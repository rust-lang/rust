#ifndef RUST_UTIL_H
#define RUST_UTIL_H

#include <limits.h>
#include "rust_task.h"
#include "rust_env.h"

extern struct type_desc str_body_tydesc;

// Inline fn used regularly elsewhere.

static inline size_t
next_power_of_two(size_t s)
{
    size_t tmp = s - 1;
    tmp |= tmp >> 1;
    tmp |= tmp >> 2;
    tmp |= tmp >> 4;
    tmp |= tmp >> 8;
    tmp |= tmp >> 16;
#ifdef _LP64
    tmp |= tmp >> 32;
#endif
    return tmp + 1;
}

// Rounds |size| to the nearest |alignment|. Invariant: |alignment| is a power
// of two.
template<typename T>
static inline T
align_to(T size, size_t alignment) {
    assert(alignment);
    T x = (T)(((uintptr_t)size + alignment - 1) & ~(alignment - 1));
    return x;
}

// Interior vectors (rust-user-code level).

struct
rust_vec
{
    size_t fill;    // in bytes; if zero, heapified
    size_t alloc;   // in bytes
    uint8_t data[0];
};

struct
rust_vec_box
{
    rust_opaque_box header;
    rust_vec body;
};

template <typename T>
inline size_t vec_size(size_t elems) {
    return sizeof(rust_vec_box) + sizeof(T) * elems;
}

template <typename T>
inline T *
vec_data(rust_vec *v) {
    return reinterpret_cast<T*>(v->data);
}

inline void reserve_vec_exact(rust_task* task, rust_vec_box** vpp,
                              size_t size) {
    if (size > (*vpp)->body.alloc) {
        *vpp = (rust_vec_box*)task->kernel
            ->realloc(*vpp, size + sizeof(rust_vec_box));
        (*vpp)->body.alloc = size;
    }
}

inline void reserve_vec(rust_task* task, rust_vec_box** vpp, size_t size) {
    reserve_vec_exact(task, vpp, next_power_of_two(size));
}

typedef rust_vec_box rust_str;

inline rust_str *
make_str(rust_kernel* kernel, const char* c, size_t strlen,
         const char* name) {
    size_t str_fill = strlen + 1;
    size_t str_alloc = str_fill;
    rust_str *str = (rust_str *)
        kernel->malloc(vec_size<char>(str_fill), name);
    str->header.td = &str_body_tydesc;
    str->body.fill = str_fill;
    str->body.alloc = str_alloc;
    memcpy(&str->body.data, c, strlen);
    str->body.data[strlen] = '\0';
    return str;
}

inline rust_vec_box *
make_str_vec(rust_kernel* kernel, size_t nstrs, char **strs) {
    rust_vec_box *v = (rust_vec_box *)
        kernel->malloc(vec_size<rust_vec_box*>(nstrs),
                       "str vec interior");
    // FIXME: should have a real td (Issue #2639)
    v->header.td = NULL;
    v->body.fill = v->body.alloc = sizeof(rust_vec_box*) * nstrs;
    for (size_t i = 0; i < nstrs; ++i) {
        rust_str *str = make_str(kernel, strs[i],
                                 strlen(strs[i]),
                                 "str");
        ((rust_str**)&v->body.data)[i] = str;
    }
    return v;
}

inline size_t get_box_size(size_t body_size, size_t body_align) {
    size_t header_size = sizeof(rust_opaque_box);
    // FIXME: This alignment calculation is suspicious. Is it right?
    size_t total_size = align_to(header_size, body_align) + body_size;
    return total_size;
}

// Initialization helpers for ISAAC RNG

inline void isaac_seed(rust_kernel* kernel, uint8_t* dest)
{
    size_t size = sizeof(ub4) * RANDSIZ;
#ifdef __WIN32__
    HCRYPTPROV hProv;
    kernel->win32_require
        (_T("CryptAcquireContext"),
         CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL,
                             CRYPT_VERIFYCONTEXT|CRYPT_SILENT));
    kernel->win32_require
        (_T("CryptGenRandom"), CryptGenRandom(hProv, size, (BYTE*) dest));
    kernel->win32_require
        (_T("CryptReleaseContext"), CryptReleaseContext(hProv, 0));
#else
    int fd = open("/dev/urandom", O_RDONLY);
    assert(fd > 0);
    assert(read(fd, dest, size) == (int) size);
    assert(close(fd) == 0);
#endif
}

inline void
isaac_init(rust_kernel *kernel, randctx *rctx, rust_vec_box* user_seed)
{
    memset(rctx, 0, sizeof(randctx));

    char *env_seed = kernel->env->rust_seed;
    if (user_seed != NULL) {
        // ignore bytes after the required length
        size_t seed_len = user_seed->body.fill < sizeof(rctx->randrsl)
            ? user_seed->body.fill : sizeof(rctx->randrsl);
        memcpy(&rctx->randrsl, user_seed->body.data, seed_len);
    } else if (env_seed != NULL) {
        ub4 seed = (ub4) atoi(env_seed);
        for (size_t i = 0; i < RANDSIZ; i ++) {
            memcpy(&rctx->randrsl[i], &seed, sizeof(ub4));
            seed = (seed + 0x7ed55d16) + (seed << 12);
        }
    } else {
        isaac_seed(kernel, (uint8_t*) &rctx->randrsl);
    }

    randinit(rctx, 1);
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

#endif
