// Routines useful when debugging the Rust runtime.

#include "rust_debug.h"
#include "rust_internal.h"

#include <iostream>
#include <string>
#include <sstream>
#include <stdint.h>

#if defined(__APPLE__) || defined(__linux__)
#define HAVE_BACKTRACE
#include <execinfo.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace {

debug::flag track_origins("RUST_TRACK_ORIGINS");

}   // end anonymous namespace

namespace debug {

#ifdef HAVE_BACKTRACE
std::string
backtrace() {
    void *call_stack[128];
    int n_frames = ::backtrace(call_stack, 128);
    char **syms = backtrace_symbols(call_stack, n_frames);

    std::cerr << "n_frames: " << n_frames << std::endl;

    std::stringstream ss;
    for (int i = 0; i < n_frames; i++) {
        std::cerr << syms[i] << std::endl;
        ss << syms[i] << std::endl;
    }

    free(syms);

    return ss.str();
}
#else
std::string
backtrace() {
    std::string s;
    return s;
}
#endif

void
maybe_track_origin(rust_task *task, void *ptr) {
    if (!*track_origins)
        return;
    task->debug.origins[ptr] = backtrace();
}

void
maybe_untrack_origin(rust_task *task, void *ptr) {
    if (!*track_origins)
        return;
    task->debug.origins.erase(ptr);
}

// This function is intended to be called by the debugger.
void
dump_origin(rust_task *task, void *ptr) {
    if (!*track_origins) {
        std::cerr << "Try again with RUST_TRACK_ORIGINS=1." << std::endl;
    } else if (task->debug.origins.find(ptr) == task->debug.origins.end()) {
        std::cerr << "Pointer " << std::hex << (uintptr_t)ptr <<
                     " does not have a tracked origin." << std::endl;
    } else {
        std::cerr << "Origin of pointer " << std::hex << (uintptr_t)ptr <<
                     ":" << std::endl << task->debug.origins[ptr] <<
                     std::endl;
    }
}

}   // end namespace debug

