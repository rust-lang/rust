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

    ptrdiff_t activate_glue_off;
    ptrdiff_t pad;
    ptrdiff_t pad2;
    ptrdiff_t gc_glue_off;
    ptrdiff_t pad3;

public:

    size_t n_rust_syms;
    size_t n_c_syms;
    size_t n_libs;

    // Crates are immutable, constructed by the compiler.

    uintptr_t get_image_base() const;
    ptrdiff_t get_relocation_diff() const;
    uintptr_t get_gc_glue() const;

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
    class lib :
        public rc_base<lib>, public dom_owned<lib>
    {
        uintptr_t handle;
    public:
        rust_dom *dom;
        lib(rust_dom *dom, char const *name);
        uintptr_t get_handle();
        ~lib();
    };

    class c_sym :
        public rc_base<c_sym>, public dom_owned<c_sym>
    {
        uintptr_t val;
        lib *library;
    public:
        rust_dom *dom;
        c_sym(rust_dom *dom, lib *library, char const *name);
        uintptr_t get_val();
        ~c_sym();
    };

    class rust_sym :
        public rc_base<rust_sym>, public dom_owned<rust_sym>
    {
        uintptr_t val;
        c_sym *crate_sym;
    public:
        rust_dom *dom;
        rust_sym(rust_dom *dom, rust_crate const *curr_crate,
                 c_sym *crate_sym, char const **path);
        uintptr_t get_val();
        ~rust_sym();
    };

    lib *get_lib(size_t n, char const *name);
    c_sym *get_c_sym(size_t n, lib *library, char const *name);
    rust_sym *get_rust_sym(size_t n,
                           rust_dom *dom,
                           rust_crate const *curr_crate,
                           c_sym *crate_sym,
                           char const **path);
    type_desc *get_type_desc(size_t size,
                             size_t align,
                             size_t n_descs,
                             type_desc const **descs);

private:

    rust_sym **rust_syms;
    c_sym **c_syms;
    lib **libs;
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
    struct mem_reader
    {
        rust_crate::mem_area &mem;
        bool ok;
        uintptr_t pos;

        bool is_ok();
        bool at_end();
        void fail();
        void reset();
        mem_reader(rust_crate::mem_area &m);
        size_t tell_abs();
        size_t tell_off();
        void seek_abs(uintptr_t p);
        void seek_off(uintptr_t p);

        template<typename T>
        void get(T &out) {
            if (pos < mem.base
                || pos >= mem.lim
                || pos + sizeof(T) > mem.lim)
                ok = false;
            if (!ok)
                return;
            out = *((T*)(pos));
            pos += sizeof(T);
            ok &= !at_end();
            I(mem.dom, at_end() || (mem.base <= pos && pos < mem.lim));
        }

        template<typename T>
        void get_uleb(T &out) {
            out = T(0);
            for (size_t i = 0; i < sizeof(T) && ok; ++i) {
                uint8_t byte = 0;
                get(byte);
                out <<= 7;
                out |= byte & 0x7f;
                if (!(byte & 0x80))
                    break;
            }
            I(mem.dom, at_end() || (mem.base <= pos && pos < mem.lim));
        }

        template<typename T>
        void adv_sizeof(T &) {
            adv(sizeof(T));
        }

        bool adv_zstr(size_t sz);
        bool get_zstr(char const *&c, size_t &sz);
        void adv(size_t amt);
    };

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

    class
    abbrev_reader : public mem_reader
    {
        ptr_vec<abbrev> abbrevs;
    public:
        abbrev_reader(rust_crate::mem_area &abbrev_mem);
        abbrev *get_abbrev(size_t i);
        bool step_attr_form_pair(uintptr_t &attr, uintptr_t &form);
        ~abbrev_reader();
    };

    rust_dom *dom;
    size_t idx;
    rust_crate const *crate;

    rust_crate::mem_area abbrev_mem;
    abbrev_reader abbrevs;

    rust_crate::mem_area die_mem;

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

    struct die_reader;

    struct
    die
    {
        die_reader *rdr;
        uintptr_t off;
        abbrev *ab;
        bool using_rdr;

        die(die_reader *rdr, uintptr_t off);
        bool is_null() const;
        bool has_children() const;
        dw_tag tag() const;
        bool start_attrs() const;
        bool step_attr(attr &a) const;
        bool find_str_attr(dw_at at, char const *&c);
        bool find_num_attr(dw_at at, uintptr_t &n);
        bool is_transparent();
        bool find_child_by_name(char const *c, die &child,
                                bool exact=false);
        bool find_child_by_tag(dw_tag tag, die &child);
        die next() const;
        die next_sibling() const;
    };

    struct
    rdr_sess
    {
        die_reader *rdr;
        rdr_sess(die_reader *rdr);
        ~rdr_sess();
    };

    struct
    die_reader : public mem_reader
    {
        abbrev_reader &abbrevs;
        uint32_t cu_unit_length;
        uintptr_t cu_base;
        uint16_t dwarf_vers;
        uint32_t cu_abbrev_off;
        uint8_t sizeof_addr;
        bool in_use;

        die first_die();
        void dump();
        die_reader(rust_crate::mem_area &die_mem,
                   abbrev_reader &abbrevs);
        ~die_reader();
    };
    die_reader dies;
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
