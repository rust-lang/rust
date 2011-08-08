#include "rust_test_runtime.h"

rust_test_runtime::rust_test_runtime() {
}

rust_test_runtime::~rust_test_runtime() {
}

#define DOMAINS 32
#define TASKS 32

void
rust_domain_test::worker::run() {
    for (int i = 0; i < TASKS; i++) {
        kernel->create_task(NULL, "child");
    }
    //sync::sleep(rand(&handle->rctx) % 1000);
}

bool
rust_domain_test::run() {
    rust_env env;
    rust_srv srv(&env);
    rust_kernel kernel(&srv, 1);

    array_list<worker *> workers;
    for (int i = 0; i < DOMAINS; i++) {
        worker *worker = new rust_domain_test::worker (&kernel);
        workers.append(worker);
        worker->start();
    }

    // We don't join the worker threads here in order to simulate ad-hoc
    // termination of domains. If we join_all_domains before all domains
    // are actually spawned, this could crash, thus the reason for the
    // sleep below.

    sync::sleep(100);
    return true;
}

void task_entry() {
    printf("task entry\n");
}

void
rust_task_test::worker::run() {
    rust_task_id root_id = kernel->create_task(NULL, "main");
    rust_task *root_task = kernel->get_task_by_id(root_id);
    root_task->start((uintptr_t)&task_entry, (uintptr_t)NULL);
    root_task->sched->start_main_loop();
}

bool
rust_task_test::run() {
    rust_env env;
    rust_srv srv(&env);
    rust_kernel kernel(&srv, 1);

    array_list<worker *> workers;
    for (int i = 0; i < DOMAINS; i++) {
        worker *worker = new rust_task_test::worker (&kernel, this);
        workers.append(worker);
        worker->start();
    }

    //sync::sleep(rand(&kernel.sched->rctx) % 1000);
    return true;
}
