/*
 *
 */

#include "rust_internal.h"
#include "rust_srv.h"

rust_srv::rust_srv() :
    local_region(this, false),
    synchronized_region(this, true) {
    // Nop.
}

rust_srv::~rust_srv() {
//    char msg[BUF_BYTES];
//    snprintf(msg, sizeof(msg), "~rust_srv %" PRIxPTR, (uintptr_t) this);
//    log(msg);
}

void
rust_srv::free(void *p) {
    ::free(p);
}

void *
rust_srv::malloc(size_t bytes) {
    return ::malloc(bytes);
}

void *
rust_srv::realloc(void *p, size_t bytes) {
    return ::realloc(p, bytes);
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
    char buf[BUF_BYTES];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    char msg[BUF_BYTES];
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
    char buf[BUF_BYTES];
    va_list args;
    va_start(args, format);
    vsnprintf(buf, sizeof(buf), format, args);
    va_end(args);

    char msg[BUF_BYTES];
    snprintf(msg, sizeof(msg),
             "warning: '%s', at: %s:%d %s",
             expression, file, (int)line, buf);
    log(msg);
}

rust_srv *
rust_srv::clone() {
    return new rust_srv();
}
