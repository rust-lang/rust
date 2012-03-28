/*
 * Interrupt transparent queue, Schoen et. al, "On Interrupt-Transparent
 * Synchronization in an Embedded Object-Oriented Operating System", 2000.
 * enqueue() is allowed to interrupt enqueue() and dequeue(), however,
 * dequeue() is not allowed to interrupt itself.
 */

#include "../rust_globals.h"
#include "lock_free_queue.h"

lock_free_queue_node::lock_free_queue_node() : next(NULL) {

}

lock_free_queue::lock_free_queue() : _tail(this) {

}

void
lock_free_queue::enqueue(lock_free_queue_node *item) {
    lock.lock();
    item->next = (lock_free_queue_node *) NULL;
    lock_free_queue_node *last = _tail;
    _tail = item;
    while (last->next) {
        last = last->next;
    }
    last->next = item;
    lock.unlock();
}

lock_free_queue_node *
lock_free_queue::dequeue() {
    lock.lock();
    lock_free_queue_node *item = next;
    if (item && !(next = item->next)) {
        _tail = (lock_free_queue_node *) this;
        if (item->next) {
            lock_free_queue_node *lost = item->next;
            lock_free_queue_node *help;
            do {
                help = lost->next;
                enqueue(lost);
            } while ((lost = help) != (lock_free_queue_node *) NULL);
        }
    }
    lock.unlock();
    return item;
}

bool
lock_free_queue::is_empty() {
    return next == NULL;
}
