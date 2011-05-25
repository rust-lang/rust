#ifndef RUST_INTERNAL_H
#define RUST_INTERNAL_H

#define __STDC_LIMIT_MACROS 1
#define __STDC_CONSTANT_MACROS 1
#define __STDC_FORMAT_MACROS 1

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

struct rust_dom;
struct rust_task;
class rust_log;
class rust_port;
class rust_chan;
struct rust_token;
class rust_kernel;
class rust_crate;
class rust_crate_cache;

struct stk_seg;
struct type_desc;
struct frame_glue_fns;

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

// Every reference counted object should derive from this base class.

template <typename T> struct rc_base {
    intptr_t ref_count;

    void ref() {
        ++ref_count;
    }

    void deref() {
        if (--ref_count == 0) {
            delete (T*)this;
        }
    }

    rc_base();
    ~rc_base();
};

template <typename T> struct dom_owned {
    void operator delete(void *ptr) {
        ((T *)ptr)->dom->free(ptr);
    }
};

template <typename T> struct task_owned {
    void operator delete(void *ptr) {
        ((T *)ptr)->task->dom->free(ptr);
    }
};

template <typename T> struct kernel_owned {
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

template <typename T> class ptr_vec : public dom_owned<ptr_vec<T> > {
    static const size_t INIT_SIZE = 8;
    rust_dom *dom;
    size_t alloc;
    size_t fill;
    T **data;
public:
    ptr_vec(rust_dom *dom);
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
#include "rust_proxy.h"
#include "rust_kernel.h"
#include "rust_message.h"
#include "rust_dom.h"
#include "memory.h"

struct rust_timer {
    // FIXME: This will probably eventually need replacement
    // with something more sophisticated and integrated with
    // an IO event-handling library, when we have such a thing.
    // For now it's just the most basic "thread that can interrupt
    // its associated domain-thread" device, so that we have
    // *some* form of task-preemption.
    rust_dom *dom;
    uintptr_t exit_flag;

#if defined(__WIN32__)
    HANDLE thread;
#else
    pthread_attr_t attr;
    pthread_t thread;
#endif

    rust_timer(rust_dom *dom);
    ~rust_timer();
};

#include "rust_util.h"

// Crates.

template<typename T> T*
crate_rel(rust_crate const *crate, T *t) {
    return (T*)(((uintptr_t)crate) + ((ptrdiff_t)t));
}

template<typename T> T const*
crate_rel(rust_crate const *crate, T const *t) {
    return (T const*)(((uintptr_t)crate) + ((ptrdiff_t)t));
}

typedef void CDECL (*activate_glue_ty)(rust_task *);

class rust_crate {
    // The following fields are emitted by the compiler for the static
    // rust_crate object inside each compiled crate.

    ptrdiff_t image_base_off;     // (Loaded image base) - this.
    uintptr_t self_addr;          // Un-relocated addres of 'this'.

    ptrdiff_t debug_abbrev_off;   // Offset from this to .debug_abbrev.
    size_t debug_abbrev_sz;       // Size of .debug_abbrev.

    ptrdiff_t debug_info_off;     // Offset from this to .debug_info.
    size_t debug_info_sz;         // Size of .debug_info.

    ptrdiff_t pad;
    ptrdiff_t pad2;
    ptrdiff_t pad3;
    ptrdiff_t pad4;
    ptrdiff_t pad5;

public:

    size_t pad6;
    size_t pad7;
    size_t pad8;

    // Crates are immutable, constructed by the compiler.

    uintptr_t get_image_base() const;
    ptrdiff_t get_relocation_diff() const;

    struct mem_area
    {
      rust_dom *dom;
      uintptr_t base;
      uintptr_t lim;
      mem_area(rust_dom *dom, uintptr_t pos, size_t sz);
    };

    mem_area get_debug_info(rust_dom *dom) const;
    mem_area get_debug_abbrev(rust_dom *dom) const;
};


struct type_desc {
    // First part of type_desc is known to compiler.
    // first_param = &descs[1] if dynamic, null if static.
    const type_desc **first_param;
    size_t size;
    size_t align;
    uintptr_t copy_glue_off;
    uintptr_t drop_glue_off;
    uintptr_t free_glue_off;
    uintptr_t sever_glue_off;    // For GC.
    uintptr_t mark_glue_off;     // For GC.
    uintptr_t obj_drop_glue_off; // For custom destructors.
    uintptr_t is_stateful;

    // Residual fields past here are known only to runtime.
    UT_hash_handle hh;
    size_t n_descs;
    const type_desc *descs[];
};

class
rust_crate_cache : public dom_owned<rust_crate_cache>,
                   public rc_base<rust_crate_cache>
{
public:
    type_desc *get_type_desc(size_t size,
                             size_t align,
                             size_t n_descs,
                             type_desc const **descs);

private:

    type_desc *type_descs;

public:

    rust_crate const *crate;
    rust_dom *dom;
    size_t idx;

    rust_crate_cache(rust_dom *dom,
                     rust_crate const *crate);
    ~rust_crate_cache();
    void flush();
};

#include "rust_dwarf.h"

class
rust_crate_reader
{
    struct
    abbrev : dom_owned<abbrev>
    {
        rust_dom *dom;
        uintptr_t body_off;
        size_t body_sz;
        uintptr_t tag;
        uint8_t has_children;
        size_t idx;
        abbrev(rust_dom *dom, uintptr_t body_off, size_t body_sz,
               uintptr_t tag, uint8_t has_children);
    };

    rust_dom *dom;
    size_t idx;


public:

    struct
    attr
    {
        dw_form form;
        dw_at at;
        union {
            struct {
                char const *s;
                size_t sz;
            } str;
            uintptr_t num;
        } val;

        bool is_numeric() const;
        bool is_string() const;
        size_t get_ssz(rust_dom *dom) const;
        char const *get_str(rust_dom *dom) const;
        uintptr_t get_num(rust_dom *dom) const;
        bool is_unknown() const;
    };

    rust_crate_reader(rust_dom *dom, rust_crate const *crate);
};

// An alarm can be put into a wait queue and the task will be notified
// when the wait queue is flushed.

struct
rust_alarm
{
    rust_task *receiver;
    size_t idx;

    rust_alarm(rust_task *receiver);
};


typedef ptr_vec<rust_alarm> rust_wait_queue;


struct stk_seg {
    unsigned int valgrind_id;
    uintptr_t limit;
    uint8_t data[];
};

struct frame_glue_fns {
    uintptr_t mark_glue_off;
    uintptr_t drop_glue_off;
    uintptr_t reloc_glue_off;
};

struct gc_alloc {
    gc_alloc *prev;
    gc_alloc *next;
    uintptr_t ctrl_word;
    uint8_t data[];
    bool mark() {
        if (ctrl_word & 1)
            return false;
        ctrl_word |= 1;
        return true;
    }
};

#include "circular_buffer.h"
#include "rust_task.h"
#include "rust_chan.h"
#include "rust_port.h"

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
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

#endif
