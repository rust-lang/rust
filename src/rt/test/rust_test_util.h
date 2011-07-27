#ifndef RUST_TEST_UTIL_H
#define RUST_TEST_UTIL_H

class rust_test_util : public rust_test {
public:

};

class rust_array_list_test : public rust_test {
public:
    bool run();
    const char *name() {
        return "rust_array_list_test";
    }
};


class rust_synchronized_indexed_list_test : public rust_test {
public:
    rust_env env;
    rust_srv srv;
    synchronized_indexed_list<indexed_list_element<int> > list;

    rust_synchronized_indexed_list_test() :
        srv(&env)
    {
    }

    class worker : public rust_thread {
    public:
        rust_synchronized_indexed_list_test *parent;
        worker(rust_synchronized_indexed_list_test *parent) : parent(parent) {
        }
        void run();
    };
    bool run();
    const char *name() {
        return "rust_synchronized_indexed_list_test";
    }
};

#endif /* RUST_TEST_UTIL_H */
