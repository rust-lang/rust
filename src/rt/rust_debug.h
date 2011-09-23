// Routines useful when debugging the Rust runtime.

#ifndef RUST_DEBUG_H
#define RUST_DEBUG_H

#include <cstdlib>

namespace debug {

class flag {
private:
    const char *name;
    bool valid;
    bool value;

public:
    flag(const char *in_name) : name(in_name), valid(false) {}

    bool operator*() {
        // FIXME: We ought to lock this.
        if (!valid) {
            char *ev = getenv(name);
            value = ev && ev[0] != '\0' && ev[0] != '0';
            valid = true;
        }
        return value;
    }
};

}   // end namespace debug

#endif

