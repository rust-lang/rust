/*
 *
 */

#ifndef RUST_SRV_H
#define RUST_SRV_H

class rust_srv {
private:
    size_t _live_allocations;
    array_list<void *> _allocation_list;
public:
    virtual void log(char const *msg);
    virtual void fatal(char const *expression,
        char const *file,
        size_t line,
        char const *format,
        ...);
    virtual void warning(char const *expression,
        char const *file,
        size_t line,
        char const *format,
        ...);
    virtual void *malloc(size_t);
    virtual void *realloc(void *, size_t);
    virtual void free(void *);
    virtual rust_srv *clone();
    rust_srv();
    virtual ~rust_srv();
};

#endif /* RUST_SRV_H */
