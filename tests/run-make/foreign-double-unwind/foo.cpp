#include <cstdio>
#include <exception>

void println(const char* s) {
    puts(s);
    fflush(stdout);
}

struct outer_exception {};
struct inner_exception {};

extern "C" {
    void throw_cxx_exception() {
        if (std::uncaught_exception()) {
            println("throwing inner C++ exception");
            throw inner_exception();
        } else {
            println("throwing outer C++ exception");
            throw outer_exception();
        }
    }

    void cxx_catch_callback(void (*cb)()) {
        try {
            cb();
            println("unreachable: callback returns");
        } catch (outer_exception) {
            println("unreachable: caught outer exception in catch (...)");
        } catch (inner_exception) {
            println("unreachable: caught inner exception in catch (...)");
        }
    }
}
