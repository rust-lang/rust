// -*- c++ -*-
#ifndef RUST_SRV_H
#define RUST_SRV_H

#include "rust_internal.h"

class rust_srv {
public:
    rust_env *env;
    memory_region local_region;
    void log(char const *msg);
    void fatal(char const *expression,
        char const *file,
        size_t line,
        char const *format,
        ...);
    void warning(char const *expression,
        char const *file,
        size_t line,
        char const *format,
        ...);
    void free(void *);
    void *malloc(size_t);
    void *realloc(void *, size_t);
    rust_srv(rust_env *);
    rust_srv *clone();
};

#endif /* RUST_SRV_H */
