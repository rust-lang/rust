#include "rust_internal.h"

rust_task_list::rust_task_list (rust_dom *dom, const char* name) :
    dom(dom), name(name) {
    // Nop;
}

void
rust_task_list::delete_all() {
    dom->log(rust_log::TASK, "deleting all %s tasks", name);
    while (is_empty() == false) {
        rust_task *task = pop_value();
        dom->log(rust_log::TASK, "deleting task " PTR, task);
        delete task;
    }
}
