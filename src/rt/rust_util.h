#ifndef RUST_UTIL_H
#define RUST_UTIL_H

#include "rust_task.h"
#include <limits.h>

// Utility type: pointer-vector.

template <typename T>
ptr_vec<T>::ptr_vec(rust_task *task) :
    task(task),
    alloc(INIT_SIZE),
    fill(0),
    data(new (task, "ptr_vec<T>") T*[alloc])
{
    I(task->thread, data);
    DLOG(task->thread, mem, "new ptr_vec(data=0x%" PRIxPTR ") -> 0x%" PRIxPTR,
         (uintptr_t)data, (uintptr_t)this);
}

template <typename T>
ptr_vec<T>::~ptr_vec()
{
    I(task->thread, data);
    DLOG(task->thread, mem, "~ptr_vec 0x%" PRIxPTR ", data=0x%" PRIxPTR,
         (uintptr_t)this, (uintptr_t)data);
    I(task->thread, fill == 0);
    task->free(data);
}

template <typename T> T *&
ptr_vec<T>::operator[](size_t offset) {
    I(task->thread, data[offset]->idx == offset);
    return data[offset];
}

template <typename T>
void
ptr_vec<T>::push(T *p)
{
    I(task->thread, data);
    I(task->thread, fill <= alloc);
    if (fill == alloc) {
        alloc *= 2;
        data = (T **)task->realloc(data, alloc * sizeof(T*));
        I(task->thread, data);
    }
    I(task->thread, fill < alloc);
    p->idx = fill;
    data[fill++] = p;
}

template <typename T>
T *
ptr_vec<T>::pop()
{
    return data[--fill];
}

template <typename T>
T *
ptr_vec<T>::peek()
{
    return data[fill - 1];
}

template <typename T>
void
ptr_vec<T>::trim(size_t sz)
{
    I(task->thread, data);
    if (sz <= (alloc / 4) &&
        (alloc / 2) >= INIT_SIZE) {
        alloc /= 2;
        I(task->thread, alloc >= fill);
        data = (T **)task->realloc(data, alloc * sizeof(T*));
        I(task->thread, data);
    }
}

template <typename T>
void
ptr_vec<T>::swap_delete(T *item)
{
    /* Swap the endpoint into i and decr fill. */
    I(task->thread, data);
    I(task->thread, fill > 0);
    I(task->thread, item->idx < fill);
    fill--;
    if (fill > 0) {
        T *subst = data[fill];
        size_t idx = item->idx;
        data[idx] = subst;
        subst->idx = idx;
    }
}

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

// Initialization helper for ISAAC RNG

template <typename thread_or_kernel>
static inline void
isaac_init(thread_or_kernel *thread, randctx *rctx)
{
        memset(rctx, 0, sizeof(randctx));

        char *rust_seed = thread->env->rust_seed;
        if (rust_seed != NULL) {
            ub4 seed = (ub4) atoi(rust_seed);
            for (size_t i = 0; i < RANDSIZ; i ++) {
                memcpy(&rctx->randrsl[i], &seed, sizeof(ub4));
                seed = (seed + 0x7ed55d16) + (seed << 12);
            }
        } else {
#ifdef __WIN32__
            HCRYPTPROV hProv;
            thread->win32_require
                (_T("CryptAcquireContext"),
                 CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_FULL,
                                     CRYPT_VERIFYCONTEXT|CRYPT_SILENT));
            thread->win32_require
                (_T("CryptGenRandom"),
                 CryptGenRandom(hProv, sizeof(rctx->randrsl),
                                (BYTE*)(&rctx->randrsl)));
            thread->win32_require
                (_T("CryptReleaseContext"),
                 CryptReleaseContext(hProv, 0));
#else
            int fd = open("/dev/urandom", O_RDONLY);
            I(thread, fd > 0);
            I(thread,
              read(fd, (void*) &rctx->randrsl, sizeof(rctx->randrsl))
              == sizeof(rctx->randrsl));
            I(thread, close(fd) == 0);
#endif
        }

        randinit(rctx, 1);
}

// Interior vectors (rust-user-code level).

struct
rust_vec
{
    size_t fill;    // in bytes; if zero, heapified
    size_t alloc;   // in bytes
    uint8_t data[0];
};

template <typename T>
inline size_t vec_size(size_t elems) {
    return sizeof(rust_vec) + sizeof(T) * elems;
}

inline void reserve_vec(rust_task* task, rust_vec** vpp, size_t size) {
    if (size > (*vpp)->alloc) {
        size_t new_alloc = next_power_of_two(size);
        *vpp = (rust_vec*)task->kernel->realloc(*vpp, new_alloc +
                                                sizeof(rust_vec));
        (*vpp)->alloc = new_alloc;
    }
}

typedef rust_vec rust_str;

inline rust_str *
make_str(rust_kernel* kernel, const char* c, size_t strlen, const char* name) {
    size_t str_fill = strlen + 1;
    size_t str_alloc = str_fill;
    rust_str *str = (rust_str *)
        kernel->malloc(vec_size<char>(str_fill), name);
    str->fill = str_fill;
    str->alloc = str_alloc;
    memcpy(&str->data, c, strlen);
    str->data[strlen] = '\0';
    return str;
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
