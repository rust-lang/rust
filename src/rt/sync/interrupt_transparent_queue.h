#ifndef INTERRUPT_TRANSPARENT_QUEUE_H
#define INTERRUPT_TRANSPARENT_QUEUE_H

#include "spin_lock.h"

class interrupt_transparent_queue_node {
public:
    interrupt_transparent_queue_node *next;
    interrupt_transparent_queue_node();
};

class interrupt_transparent_queue : interrupt_transparent_queue_node {
    spin_lock lock;
    interrupt_transparent_queue_node *_tail;
public:
    interrupt_transparent_queue();
    void enqueue(interrupt_transparent_queue_node *item);
    interrupt_transparent_queue_node *dequeue();
    bool is_empty();
};

#endif /* INTERRUPT_TRANSPARENT_QUEUE_H */
