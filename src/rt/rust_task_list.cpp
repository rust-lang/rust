#include "rust_internal.h"

rust_task_list::rust_task_list (rust_task_thread *thread, const char* name) :
    thread(thread), name(name) {
}

void
rust_task_list::delete_all() {
    DLOG(thread, task, "deleting all %s tasks", name);
    while (is_empty() == false) {
        rust_task *task = pop_value();
        DLOG(thread, task, "deleting task " PTR, task);
        delete task;
    }
}
