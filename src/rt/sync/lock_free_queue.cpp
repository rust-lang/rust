/*
 * Interrupt transparent queue, Schoen et. al, "On Interrupt-Transparent
 * Synchronization in an Embedded Object-Oriented Operating System", 2000.
 * enqueue() is allowed to interrupt enqueue() and dequeue(), however,
 * dequeue() is not allowed to interrupt itself.
 */

#include "lock_free_queue.h"

lock_free_queue::lock_free_queue() :
    tail(this) {
}

void lock_free_queue::enqueue(lock_free_queue_node *item) {
    item->next = (lock_free_queue_node *) 0;
    lock_free_queue_node *last = tail;
    tail = item;
    while (last->next)
        last = last->next;
    last->next = item;
}

lock_free_queue_node *lockfree_queue::dequeue() {
    lock_free_queue_node *item = next;
    if (item && !(next = item->next)) {
        tail = (lock_free_queue_node *) this;
        if (item->next) {
            lock_free_queue_node *lost = item->next;
            lock_free_queue_node *help;
            do {
                help = lost->next;
                enqueue(lost);
            } while ((lost = help) != (lock_free_queue_node *) 0);
        }
    }
    return item;
}
