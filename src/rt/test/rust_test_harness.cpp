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
        if (tests[i]->run() == false) {
            printf("test: %s FAILED\n", test->name());
            pass = false;
        } else {
            printf("test: %s PASSED\n", test->name());
        }
    }
    return pass;
}

