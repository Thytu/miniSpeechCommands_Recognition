import torch
import torch.nn.functional as F

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def train(network, optimizer, train_set, train_labels, batch_size=32):
    TRAINING_SIZE = len(train_set) * batch_size
    network.train()
    correct_in_episode = 0
    episode_loss = 0

    for index, images in enumerate(train_set):
        labels = train_labels[index]

        predictions = network(images)

        loss = F.cross_entropy(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_loss += loss.item()
        correct_in_episode += get_num_correct(predictions, labels)

    return episode_loss, correct_in_episode * 100 / TRAINING_SIZE

def test(network, test_set, test_labels, batch_size=32):
    TESTING_SIZE = len(test_set) * batch_size
    network.eval()
    episode_loss = 0
    correct_in_episode = 0

    with torch.no_grad():
        for index, images in enumerate(test_set):
            labels = test_labels[index]

            predictions = network(images)
            loss = F.cross_entropy(predictions, labels)

            episode_loss = loss.item()
            correct_in_episode += get_num_correct(predictions, labels)

    return episode_loss, correct_in_episode * 100 / TESTING_SIZE