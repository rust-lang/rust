#include "../rust_internal.h"

#define COUNT 1000
#define LARGE_COUNT 10000
#define THREADS 10

bool
rust_array_list_test::run() {
    array_list<int> list;

    for (int i = 0; i < COUNT; i++) {
        list.append(i);
    }

    for (int i = 0; i < COUNT; i++) {
        CHECK (list[i] == i);
    }

    for (int i = 0; i < COUNT; i++) {
        CHECK (list.index_of(i) == i);
    }

    for (int i = 0; i < COUNT; i++) {
        CHECK (list.replace(i, -i));
        CHECK (list.replace(-i, i));
        CHECK (list.index_of(i) == i);
    }

    for (int i = COUNT - 1; i >= 0; i--) {
        CHECK (list.pop(NULL));
    }

    return true;
}

bool
rust_synchronized_indexed_list_test::run() {
    array_list<worker*> workers;

    for (int i = 0; i < THREADS; i++) {
        worker *worker =
            new rust_synchronized_indexed_list_test::worker(this);
        workers.append(worker);
    }

    for (uint32_t i = 0; i < workers.size(); i++) {
        workers[i]->start();
    }

    while(workers.is_empty() == false) {
        worker *worker;
        workers.pop(&worker);
        worker->join();
        delete worker;
    }

    size_t expected_items = LARGE_COUNT * THREADS;

    CHECK(list.length() == expected_items);

    long long sum = 0;
    for (size_t i = 0; i < list.length(); i++) {
        sum += list[i]->value;
    }

    long long expected_sum = LARGE_COUNT;
    expected_sum = expected_sum * (expected_sum - 1) / 2 * THREADS;
    CHECK (sum == expected_sum);
    return true;
}

void
rust_synchronized_indexed_list_test::worker::run() {
    for (int i = 0; i < LARGE_COUNT; i++) {
        parent->list.append(new indexed_list_element<int>(i));
    }
    return;
}
