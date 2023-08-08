#include <assert.h>
#include <stddef.h>
#include <stdio.h>

void println(const char* s) {
    puts(s);
    fflush(stdout);
}

struct exception {};
struct rust_panic {};

struct drop_check {
    bool* ok;
    ~drop_check() {
        println("~drop_check");

        if (ok)
            *ok = true;
    }
};

extern "C" {
    void rust_catch_callback(void (*cb)(), bool* rust_ok);

    void throw_cxx_exception() {
        println("throwing C++ exception");
        throw exception();
    }

    void test_cxx_exception() {
        bool rust_ok = false;
        try {
            rust_catch_callback(throw_cxx_exception, &rust_ok);
            assert(false && "unreachable");
        } catch (exception e) {
            println("caught C++ exception");
            assert(rust_ok);
            return;
        }
        assert(false && "did not catch thrown C++ exception");
    }

    void cxx_catch_callback(void (*cb)(), bool* cxx_ok) {
        drop_check x;
        x.ok = NULL;
        try {
            cb();
        } catch (rust_panic e) {
            assert(false && "shouldn't be able to catch a rust panic");
        } catch (...) {
            println("caught foreign exception in catch (...)");
            // Foreign exceptions are caught by catch (...). We only set the ok
            // flag if we successfully caught the panic. The destructor of
            // drop_check will then set the flag to true if it is executed.
            x.ok = cxx_ok;
            throw;
        }
    }
}
