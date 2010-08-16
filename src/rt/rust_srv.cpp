/*
 *
 */

#include "rust_internal.h"
#include "rust_srv.h"

#define TRACK_ALLOCATIONS

rust_srv::rust_srv() : _live_allocations(0) {
    // Nop.
}

rust_srv::~rust_srv() {
    if (_live_allocations != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "leaked memory in rust main loop (%" PRIuPTR " objects)",
                 _live_allocations);
#ifdef TRACK_ALLOCATIONS
        for (size_t i = 0; i < _allocation_list.size(); i++) {
            if (_allocation_list[i] != NULL) {
                printf("allocation 0x%" PRIxPTR " was not freed\n",
                        (uintptr_t) _allocation_list[i]);
            }
        }
#endif
        fatal(msg, __FILE__, __LINE__, "");
    }
}

void *
rust_srv::malloc(size_t bytes) {
    ++_live_allocations;
    void * val = ::malloc(bytes);
#ifdef TRACK_ALLOCATIONS
    _allocation_list.append(val);
#endif
    return val;
}

void *
rust_srv::realloc(void *p, size_t bytes) {
    if (!p) {
        _live_allocations++;
    }
    void * val = ::realloc(p, bytes);
#ifdef TRACK_ALLOCATIONS
    if (_allocation_list.replace(p, val) == false) {
        printf("realloc: ptr 0x%" PRIxPTR " is not in allocation_list\n",
               (uintptr_t) p);
        fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
#endif
    return val;
}

void
rust_srv::free(void *p) {
#ifdef TRACK_ALLOCATIONS
    if (_allocation_list.replace(p, NULL) == false) {
        printf("free: ptr 0x%" PRIxPTR " is not in allocation_list\n",
               (uintptr_t) p);
        fatal("not in allocation_list", __FILE__, __LINE__, "");
    }
#endif
    if (_live_allocations < 1) {
        fatal("live_allocs < 1", __FILE__, __LINE__, "");
    }
    _live_allocations--;
    ::free(p);
}

void
rust_srv::log(char const *msg) {
    printf("rt: %s\n", msg);
}

void
rust_srv::fatal(const char *expression,
    const char *file,
    size_t line,
    const char *format,
    ...) {
    char buf[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    char msg[1024];
    snprintf(msg, sizeof(msg),
             "fatal, '%s' failed, %s:%d %s",
             expression, file, (int)line, buf);
    log(msg);
    exit(1);
}

void
rust_srv::warning(char const *expression,
    char const *file,
    size_t line,
    const char *format,
    ...) {
    char buf[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    char msg[1024];
    snprintf(msg, sizeof(msg),
             "warning: '%s', at: %s:%d %s",
             expression, file, (int)line, buf);
    log(msg);
}

rust_srv *
rust_srv::clone() {
    return new rust_srv();
}
