// Routines useful when debugging the Rust runtime.

#include "rust_abi.h"
#include "rust_debug.h"
#include "rust_internal.h"

#include <iostream>
#include <string>
#include <sstream>
#include <stdint.h>

namespace {

debug::flag track_origins("RUST_TRACK_ORIGINS");

}   // end anonymous namespace

namespace debug {

void
maybe_track_origin(rust_task *task, void *ptr) {
    if (!*track_origins)
        return;
    task->debug.origins[ptr] =
        stack_walk::symbolicate(stack_walk::backtrace());
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

