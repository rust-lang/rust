#include "rust_internal.h"

rust_task_list::rust_task_list (rust_scheduler *sched, const char* name) :
    sched(sched), name(name) {
}

void
rust_task_list::delete_all() {
    DLOG(sched, task, "deleting all %s tasks", name);
    while (is_empty() == false) {
        rust_task *task = pop_value();
        DLOG(sched, task, "deleting task " PTR, task);
        delete task;
    }
}
