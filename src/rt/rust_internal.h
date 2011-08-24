#ifndef RUST_INTERNAL_H
#define RUST_INTERNAL_H

#ifndef GLOBALS_H
// these are defined in two files, and GCC complains.
#define __STDC_LIMIT_MACROS 1
#define __STDC_CONSTANT_MACROS 1
#define __STDC_FORMAT_MACROS 1
#endif

#define ERROR 0

#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdarg.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>

#include "rust.h"
#include "rand.h"
#include "uthash.h"
#include "rust_env.h"

#if defined(__WIN32__)
extern "C" {
#include <windows.h>
#include <tchar.h>
#include <wincrypt.h>
}
#elif defined(__GNUC__)
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#include <errno.h>
#include <dirent.h>
#else
#error "Platform not supported."
#endif

#include "util/array_list.h"
#include "util/indexed_list.h"
#include "util/synchronized_indexed_list.h"
#include "util/hash_map.h"
#include "sync/sync.h"
#include "sync/timer.h"
#include "sync/lock_and_signal.h"
#include "sync/lock_free_queue.h"

struct rust_scheduler;
struct rust_task;
class rust_log;
class rust_port;
class rust_chan;
struct rust_token;
class rust_kernel;
class rust_crate_cache;

struct stk_seg;
struct type_desc;
struct frame_glue_fns;

typedef intptr_t rust_task_id;
typedef intptr_t rust_port_id;

// Corresponds to the rust chan (currently _chan) type.
struct chan_handle {
    rust_task_id task;
    rust_port_id port;
};

#ifndef __i386__
#error "Target CPU not supported."
#endif

#define I(dom, e) ((e) ? (void)0 : \
         (dom)->srv->fatal(#e, __FILE__, __LINE__, ""))

#define W(dom, e, s, ...) ((e) ? (void)0 : \
         (dom)->srv->warning(#e, __FILE__, __LINE__, s, ## __VA_ARGS__))

#define A(dom, e, s, ...) ((e) ? (void)0 : \
         (dom)->srv->fatal(#e, __FILE__, __LINE__, s, ## __VA_ARGS__))

#define K(srv, e, s, ...) ((e) ? (void)0 : \
         srv->fatal(#e, __FILE__, __LINE__, s, ## __VA_ARGS__))

#define PTR "0x%" PRIxPTR

// This drives our preemption scheme.

static size_t const TIME_SLICE_IN_MS = 10;

// Since every refcounted object is > 4 bytes, any refcount with any of the
// top two bits set is invalid. We reserve a particular bit pattern in this
// set for indicating objects that are "constant" as far as the memory model
// knows.

static intptr_t const CONST_REFCOUNT = 0x7badface;

// This accounts for logging buffers.

static size_t const BUF_BYTES = 2048;

// Every reference counted object should use this macro and initialize
// ref_count.

#define RUST_REFCOUNTED(T) \
  RUST_REFCOUNTED_WITH_DTOR(T, delete (T*)this)
#define RUST_REFCOUNTED_WITH_DTOR(T, dtor) \
  intptr_t ref_count;      \
  void ref() { ++ref_count; } \
  void deref() { if (--ref_count == 0) { dtor; } }

#define RUST_ATOMIC_REFCOUNT()                                          \
    private:                                                            \
    intptr_t ref_count;                                                 \
public:                                                                 \
 void ref() {                                                           \
     intptr_t old = sync::increment(ref_count);                         \
     assert(old > 0);                                                   \
 }                                                                      \
 void deref() { if(0 == sync::decrement(ref_count)) { delete this; } }

template <typename T> struct task_owned {
    inline void *operator new(size_t size, rust_task *task, const char *tag);

    inline void *operator new[](size_t size, rust_task *task,
                                const char *tag);

    inline void *operator new(size_t size, rust_task &task, const char *tag);

    inline void *operator new[](size_t size, rust_task &task,
                                const char *tag);

    void operator delete(void *ptr) {
        ((T *)ptr)->task->free(ptr);
    }
};

template<class T>
class smart_ptr {
    T *p;

public:
    smart_ptr() : p(NULL) {};
    smart_ptr(T *p) : p(p) { if(p) { p->ref(); } }
    smart_ptr(const smart_ptr &sp) : p(sp.p) {
        if(p) { p->ref(); }
    }

    ~smart_ptr() {
        if(p) {
            p->deref();
        }
    }

    T *operator=(T* p) {
        if(this->p) {
            this->p->deref();
        }
        if(p) {
            p->ref();
        }
        this->p = p;

        return p;
    }

    T *operator->() const { return p; };

    operator T*() const { return p; }
};

template <typename T> struct kernel_owned {
    inline void *operator new(size_t size, rust_kernel *kernel,
                              const char *tag);

    void operator delete(void *ptr) {
        ((T *)ptr)->kernel->free(ptr);
    }
};

template <typename T> struct region_owned {
    void operator delete(void *ptr) {
        ((T *)ptr)->region->free(ptr);
    }
};

#include "rust_task_list.h"

// A cond(ition) is something we can block on. This can be a channel
// (writing), a port (reading) or a task (waiting).

struct rust_cond { };

// Helper class used regularly elsewhere.

template <typename T> class ptr_vec : public task_owned<ptr_vec<T> > {
    static const size_t INIT_SIZE = 8;
    rust_task *task;
    size_t alloc;
    size_t fill;
    T **data;
public:
    ptr_vec(rust_task *task);
    ~ptr_vec();

    size_t length() {
        return fill;
    }

    bool is_empty() {
        return fill == 0;
    }

    T *& operator[](size_t offset);
    void push(T *p);
    T *pop();
    T *peek();
    void trim(size_t fill);
    void swap_delete(T* p);
};

#include "memory_region.h"
#include "rust_srv.h"
#include "rust_log.h"
#include "rust_kernel.h"
#include "rust_scheduler.h"

struct rust_timer {
    // FIXME: This will probably eventually need replacement
    // with something more sophisticated and integrated with
    // an IO event-handling library, when we have such a thing.
    // For now it's just the most basic "thread that can interrupt
    // its associated domain-thread" device, so that we have
    // *some* form of task-preemption.
    rust_scheduler *sched;
    uintptr_t exit_flag;

#if defined(__WIN32__)
    HANDLE thread;
#else
    pthread_attr_t attr;
    pthread_t thread;
#endif

    rust_timer(rust_scheduler *sched);
    ~rust_timer();
};

#include "rust_util.h"

typedef void CDECL (glue_fn)(void *, rust_task *, void *,
                             const type_desc **, void *);
typedef void CDECL (cmp_glue_fn)(void *, rust_task *, void *,
                                 const type_desc **,
                                 void *, void *, int8_t);

struct rust_shape_tables {
    uint8_t *tags;
    uint8_t *resources;
};

struct type_desc {
    // First part of type_desc is known to compiler.
    // first_param = &descs[1] if dynamic, null if static.
    const type_desc **first_param;
    size_t size;
    size_t align;
    glue_fn *take_glue;
    glue_fn *drop_glue;
    glue_fn *free_glue;
    void *unused;
    glue_fn *sever_glue;    // For GC.
    glue_fn *mark_glue;     // For GC.
    uintptr_t is_stateful;
    cmp_glue_fn *cmp_glue;
    const uint8_t *shape;
    const rust_shape_tables *shape_tables;
    uintptr_t n_params;
    uintptr_t n_obj_params;

    // Residual fields past here are known only to runtime.
    UT_hash_handle hh;
    size_t n_descs;
    const type_desc *descs[];
};

#include "circular_buffer.h"
#include "rust_task.h"
#include "rust_chan.h"
#include "rust_port.h"
#include "memory.h"

extern "C" CDECL void
port_recv(rust_task *task, uintptr_t *dptr, rust_port *port);

#include "test/rust_test_harness.h"
#include "test/rust_test_util.h"
#include "test/rust_test_runtime.h"

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

#endif
