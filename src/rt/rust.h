#ifndef RUST_H
#define RUST_H

/*
 * Include this file after you've defined the ISO C9x stdint
 * types (size_t, uint8_t, uintptr_t, etc.)
 */

#ifdef __i386__
// 'cdecl' ABI only means anything on i386
#ifdef __WIN32__
#define CDECL __cdecl
#else
#define CDECL __attribute__((cdecl))
#endif
#else
#define CDECL
#endif

struct rust_srv {
    size_t live_allocs;

    virtual void log(char const *);
    virtual void fatal(char const *, char const *, size_t);
    virtual void *malloc(size_t);
    virtual void *realloc(void *, size_t);
    virtual void free(void *);
    virtual rust_srv *clone();

    rust_srv();
    virtual ~rust_srv();
};

inline void *operator new(size_t size, rust_srv *srv)
{
    return srv->malloc(size);
}

/*
 * Local Variables:
 * fill-column: 78;
 * indent-tabs-mode: nil
 * c-basic-offset: 4
 * buffer-file-coding-system: utf-8-unix
 * compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
 * End:
 */

#endif /* RUST_H */
