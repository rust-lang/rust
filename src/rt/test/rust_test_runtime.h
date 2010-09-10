#include "../rust_internal.h"

#ifndef RUST_TEST_RUNTIME_H
#define RUST_TEST_RUNTIME_H

class rust_test_runtime {
public:
    rust_test_runtime();
    virtual ~rust_test_runtime();
};


class rust_domain_test : public rust_test {
public:
    class worker : public rust_thread {
        public:
        rust_kernel *kernel;
        worker(rust_kernel *kernel) : kernel(kernel) {
            // Nop.
        }
        void run();
    };
    bool run();
    const char *name() {
        return "rust_domain_test";
    }
};

class rust_task_test : public rust_test {
public:
    rust_test_suite *suite;
    rust_task_test(rust_test_suite *suite) : suite(suite) {
        // Nop.
    }
    class worker : public rust_thread {
        public:
        rust_kernel *kernel;
        rust_task_test *parent;
        worker(rust_kernel *kernel, rust_task_test *parent) :
            kernel(kernel), parent(parent) {
            // Nop.
        }
        void run();
    };
    bool run();
    const char *name() {
        return "rust_task_test";
    }
};

#endif /* RUST_TEST_RUNTIME_H */
