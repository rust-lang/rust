
#ifndef RUST_TASK_QUEUE_H
#define RUST_TASK_QUEUE_H

#include <cstddef>

class rust_task;

class rust_task_iterator {
 public:
    rust_task_iterator(rust_task *h, size_t m);
    
    inline bool hasNext() {
        return count < max;
    }

    rust_task *next();
    
 private:
    size_t count;
    rust_task * cur;
    size_t max;
};

class rust_task_queue {
 public:

    rust_task_queue();

    inline rust_task_iterator iterator()
    {
        return rust_task_iterator(head, sz);
    }
    
    inline size_t size() {
        return sz;
    }

    rust_task *next();
    void insert(rust_task *elem);
    void remove(rust_task *elem);
    
 private:
    
    rust_task *head;
    size_t sz;

};


#endif
