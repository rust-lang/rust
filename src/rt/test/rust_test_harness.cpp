#include "../rust_internal.h"

bool
rust_test::run() {
    return false;
}

const char *
rust_test::name() {
    return "untitled";
}

rust_test_suite::rust_test_suite() {
    tests.append(new rust_domain_test());
    tests.append(new rust_task_test(this));
    tests.append(new rust_array_list_test());
    tests.append(new rust_synchronized_indexed_list_test());
}

rust_test_suite::~rust_test_suite() {

}

bool
rust_test_suite::run() {
    bool pass = true;
    for (size_t i = 0; i < tests.size(); i++) {
        rust_test *test = tests[i];
        printf("test: %s running ... \n", test->name());
        timer timer;
        bool result = tests[i]->run();
        printf("test: %s %s %.2f ms\n", test->name(),
               result ? "PASS" : "FAIL", timer.elapsed_ms());
        if (result == false) {
            pass = false;
        }
    }
    return pass;
}

