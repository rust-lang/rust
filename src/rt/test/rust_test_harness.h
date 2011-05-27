#ifndef RUST_TEST_HARNESS_H
#define RUST_TEST_HARNESS_H

#define CHECK(x) if ((x) == false)                               \
    { printf("condition: %s failed at file: %s, line: %d\n", #x, \
             __FILE__, __LINE__ ); return false; }

class rust_test {
public:
    virtual bool run();
    virtual const char *name();
};

class rust_test_suite : public rust_test {
public:
    array_list<rust_test*> tests;
    rust_test_suite();
    virtual ~rust_test_suite();
    bool run();
};

#endif /* RUST_TEST_HARNESS_H */
