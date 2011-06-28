// -*- c++ -*-
#ifndef RUST_TASK_LIST_H
#define RUST_TASK_LIST_H

/**
 * Used to indicate the state of a rust task.
 */
class rust_task_list : public indexed_list<rust_task>,
                       public kernel_owned<rust_task_list> {
public:
    rust_scheduler *sched;
    const char* name;
    rust_task_list (rust_scheduler *sched, const char* name);
    void delete_all();
};

#endif /* RUST_TASK_LIST_H */
