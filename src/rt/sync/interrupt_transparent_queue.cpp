/*
 * Interrupt transparent queue, Schoen et. al, "On Interrupt-Transparent
 * Synchronization in an Embedded Object-Oriented Operating System", 2000.
 * enqueue() is allowed to interrupt enqueue() and dequeue(), however,
 * dequeue() is not allowed to interrupt itself.
 */

#include "../globals.h"
#include "interrupt_transparent_queue.h"

interrupt_transparent_queue_node::interrupt_transparent_queue_node() :
    next(NULL) {

}

interrupt_transparent_queue::interrupt_transparent_queue() : _tail(this) {

}

void
interrupt_transparent_queue::enqueue(interrupt_transparent_queue_node *item) {
    lock.lock();
    item->next = (interrupt_transparent_queue_node *) NULL;
    interrupt_transparent_queue_node *last = _tail;
    _tail = item;
    while (last->next) {
        last = last->next;
    }
    last->next = item;
    lock.unlock();
}

interrupt_transparent_queue_node *
interrupt_transparent_queue::dequeue() {
    lock.lock();
    interrupt_transparent_queue_node *item = next;
    if (item && !(next = item->next)) {
        _tail = (interrupt_transparent_queue_node *) this;
        if (item->next) {
            interrupt_transparent_queue_node *lost = item->next;
            interrupt_transparent_queue_node *help;
            do {
                help = lost->next;
                enqueue(lost);
            } while ((lost = help) !=
                     (interrupt_transparent_queue_node *) NULL);
        }
    }
    lock.unlock();
    return item;
}

bool
interrupt_transparent_queue::is_empty() {
    return next == NULL;
}
