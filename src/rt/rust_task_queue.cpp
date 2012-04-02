
#include "rust_task_queue.h"
#include "rust_internal.h"

rust_task_iterator::rust_task_iterator(rust_task * h, size_t m) :
    count(0), cur(h), max(m)
{
}

rust_task *
rust_task_iterator::next() {
    count++;
    rust_task *ret = cur;
    cur = cur->next;
    return ret;
}

rust_task_queue::rust_task_queue() : head(NULL), sz(0)
{
}

rust_task *
rust_task_queue::next() {
    rust_task *ret = head;
    head = head->next;
    return ret;
}

void
rust_task_queue::insert(rust_task *elem) {
    if (++sz == 1) {
        head = elem;
        head->next = head;
        head->prev = head;
    } else {
        elem->prev = head->prev;
        elem->next =  head;
        head->prev->next = elem;
        head->prev = elem;
    }
}

void
rust_task_queue::remove(rust_task *elem) {
    if (sz == 0 || elem->next == NULL || elem->prev == NULL)
        return;
    if (--sz == 0)
        head = NULL;
    else {
        if (elem == head)
            head = elem->next;
        elem->next->prev = elem->prev;
        elem->prev->next = elem->next;
    }
    elem->next = NULL;
    elem->prev = NULL;
}
