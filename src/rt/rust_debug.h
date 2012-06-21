// Routines useful when debugging the Rust runtime.

#ifndef RUST_DEBUG_H
#define RUST_DEBUG_H

#include <map>
#include <string>
#include <cstdlib>

#ifndef _WIN32

#include <signal.h>
#define BREAKPOINT_AWESOME                      \
    do {                                        \
        signal(SIGTRAP, SIG_IGN);               \
        raise(SIGTRAP);                         \
    } while (0)

#else
#define BREAKPOINT_AWESOME
#endif

struct rust_task;

namespace debug {

class flag {
private:
    const char *name;
    bool valid;
    bool value;

public:
    flag(const char *in_name) : name(in_name), valid(false) {}

    bool operator*() {
        // FIXME (#2689): We ought to lock this.
        if (!valid) {
            char *ev = getenv(name);
            value = ev && ev[0] != '\0' && ev[0] != '0';
            valid = true;
        }
        return value;
    }
};

class task_debug_info {
public:
    std::map<void *,std::string> origins;
};

std::string backtrace();

void maybe_track_origin(rust_task *task, void *ptr);
void maybe_untrack_origin(rust_task *task, void *ptr);

// This function is intended to be called by the debugger.
void dump_origin(rust_task *task, void *ptr);

}   // end namespace debug

#endif

