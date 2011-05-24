#include "rust_test_runtime.h"

rust_test_runtime::rust_test_runtime() {
    // TODO Auto-generated constructor stub
}

rust_test_runtime::~rust_test_runtime() {
    // TODO Auto-generated destructor stub
}

#define DOMAINS 32
#define TASKS 32

void
rust_domain_test::worker::run() {
    rust_handle<rust_dom> *handle = kernel->create_domain("test");
    for (int i = 0; i < TASKS; i++) {
        handle->referent()->create_task(NULL, "child");
    }
    sync::random_sleep(1000);
    kernel->destroy_domain(handle->_referent);
}

bool
rust_domain_test::run() {
    rust_srv srv;
    rust_kernel kernel(&srv);

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
    kernel.join_all_domains();
    return true;
}

void task_entry() {
    printf("task entry\n");
}

void
rust_task_test::worker::run() {
    rust_handle<rust_dom> *handle =
        kernel->create_domain("test");
    rust_dom *domain = handle->referent();
    domain->root_task->start((uintptr_t)&task_entry, (uintptr_t)NULL, 0);
    domain->start_main_loop();
    kernel->destroy_domain(domain);
}

bool
rust_task_test::run() {
    rust_srv srv;
    rust_kernel kernel(&srv);

    array_list<worker *> workers;
    for (int i = 0; i < DOMAINS; i++) {
        worker *worker = new rust_task_test::worker (&kernel, this);
        workers.append(worker);
        worker->start();
    }

    sync::random_sleep(1000);
    kernel.join_all_domains();
    return true;
}
