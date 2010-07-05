#ifndef RUST_INTERNAL_H
#define RUST_INTERNAL_H

#define __STDC_LIMIT_MACROS 1
#define __STDC_CONSTANT_MACROS 1
#define __STDC_FORMAT_MACROS 1

#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include <stdio.h>
#include <string.h>

#include "rust.h"

#include "rand.h"
#include "rust_log.h"
#include "uthash.h"

#if defined(__WIN32__)
extern "C" {
#include <windows.h>
#include <tchar.h>
#include <wincrypt.h>
}
#elif defined(__GNUC__)
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <pthread.h>
#include <errno.h>
#else
#error "Platform not supported."
#endif

#ifndef __i386__
#error "Target CPU not supported."
#endif

#define I(dom, e) ((e) ? (void)0 :                              \
                   (dom)->srv->fatal(#e, __FILE__, __LINE__))

struct rust_task;
struct rust_port;
class rust_chan;
struct rust_token;
struct rust_dom;
class rust_crate;
class rust_crate_cache;
class lockfree_queue;

struct stk_seg;
struct type_desc;
struct frame_glue_fns;

// This drives our preemption scheme.

static size_t const TIME_SLICE_IN_MS = 10;

// Every reference counted object should derive from this base class.

template <typename T>
struct
rc_base
{
    size_t refcnt;

    void ref() {
        ++refcnt;
    }

    void deref() {
        if (--refcnt == 0) {
            delete (T*)this;
        }
    }

  rc_base();
  ~rc_base();
};

template <typename T>
struct
dom_owned
{
    void operator delete(void *ptr) {
        ((T *)ptr)->dom->free(ptr);
    }
};

template <typename T>
struct
task_owned
{
    void operator delete(void *ptr) {
        ((T *)ptr)->task->dom->free(ptr);
    }
};


// Helper class used regularly elsewhere.

template <typename T>
class
ptr_vec : public dom_owned<ptr_vec<T> >
{
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

    T *& operator[](size_t offset);
    void push(T *p);
    T *pop();
    void trim(size_t fill);
    void swapdel(T* p);
};

struct
rust_dom
{
    // Fields known to the compiler:
    uintptr_t interrupt_flag;

    // Fields known only by the runtime:

    // NB: the root crate must remain in memory until the root of the
    // tree of domains exits. All domains within this tree have a
    // copy of this root_crate value and use it for finding utility
    // glue.
    rust_crate const *root_crate;
    rust_log _log;
    rust_srv *srv;
    // uint32_t logbits;
    ptr_vec<rust_task> running_tasks;
    ptr_vec<rust_task> blocked_tasks;
    ptr_vec<rust_task> dead_tasks;
    ptr_vec<rust_crate_cache> caches;
    randctx rctx;
    rust_task *root_task;
    rust_task *curr_task;
    int rval;
    lockfree_queue *incoming; // incoming messages from other threads

#ifndef __WIN32__
    pthread_attr_t attr;
#endif

    rust_dom(rust_srv *srv, rust_crate const *root_crate);
    ~rust_dom();

    void activate(rust_task *task);
    void log(uint32_t logbit, char const *fmt, ...);
    rust_log & get_log();
    void logptr(char const *msg, uintptr_t ptrval);
    template<typename T>
    void logptr(char const *msg, T* ptrval);
    void fail();
    void *malloc(size_t sz);
    void *calloc(size_t sz);
    void *realloc(void *data, size_t sz);
    void free(void *p);

#ifdef __WIN32__
    void win32_require(LPCTSTR fn, BOOL ok);
#endif

    rust_crate_cache *get_cache(rust_crate const *crate);
    size_t n_live_tasks();
    void add_task_to_state_vec(ptr_vec<rust_task> *v, rust_task *task);
    void remove_task_from_state_vec(ptr_vec<rust_task> *v, rust_task *task);
    const char *state_vec_name(ptr_vec<rust_task> *v);

    void reap_dead_tasks();
    rust_task *sched();
};

inline void *operator new(size_t sz, void *mem) {
    return mem;
}

inline void *operator new(size_t sz, rust_dom *dom) {
    return dom->malloc(sz);
}

inline void *operator new[](size_t sz, rust_dom *dom) {
    return dom->malloc(sz);
}

inline void *operator new(size_t sz, rust_dom &dom) {
    return dom.malloc(sz);
}

inline void *operator new[](size_t sz, rust_dom &dom) {
    return dom.malloc(sz);
}

struct
rust_timer
{
    // FIXME: This will probably eventually need replacement
    // with something more sophisticated and integrated with
    // an IO event-handling library, when we have such a thing.
    // For now it's just the most basic "thread that can interrupt
    // its associated domain-thread" device, so that we have
    // *some* form of task-preemption.
    rust_dom &dom;
    uintptr_t exit_flag;

#if defined(__WIN32__)
    HANDLE thread;
#else
    pthread_attr_t attr;
    pthread_t thread;
#endif

    rust_timer(rust_dom &dom);
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

class
rust_crate
{
    // The following fields are emitted by the compiler for the static
    // rust_crate object inside each compiled crate.

    ptrdiff_t image_base_off;     // (Loaded image base) - this.
    uintptr_t self_addr;          // Un-relocated addres of 'this'.

    ptrdiff_t debug_abbrev_off;   // Offset from this to .debug_abbrev.
    size_t debug_abbrev_sz;       // Size of .debug_abbrev.

    ptrdiff_t debug_info_off;     // Offset from this to .debug_info.
    size_t debug_info_sz;         // Size of .debug_info.

    ptrdiff_t activate_glue_off;
    ptrdiff_t yield_glue_off;
    ptrdiff_t unwind_glue_off;
    ptrdiff_t gc_glue_off;
    ptrdiff_t exit_task_glue_off;

public:

    size_t n_rust_syms;
    size_t n_c_syms;
    size_t n_libs;

    // Crates are immutable, constructed by the compiler.

    uintptr_t get_image_base() const;
    ptrdiff_t get_relocation_diff() const;
    activate_glue_ty get_activate_glue() const;
    uintptr_t get_yield_glue() const;
    uintptr_t get_unwind_glue() const;
    uintptr_t get_gc_glue() const;
    uintptr_t get_exit_task_glue() const;

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
                uint8_t byte;
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


// A cond(ition) is something we can block on. This can be a channel
// (writing), a port (reading) or a task (waiting).

struct
rust_cond
{
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

struct
rust_task : public rc_base<rust_task>,
            public dom_owned<rust_task>,
            public rust_cond
{
    // Fields known to the compiler.
    stk_seg *stk;
    uintptr_t runtime_sp;      // Runtime sp while task running.
    uintptr_t rust_sp;         // Saved sp when not running.
    gc_alloc *gc_alloc_chain;  // Linked list of GC allocations.
    rust_dom *dom;
    rust_crate_cache *cache;

    // Fields known only to the runtime.
    ptr_vec<rust_task> *state;
    rust_cond *cond;
    uintptr_t* dptr;           // Rendezvous pointer for send/recv.
    rust_task *supervisor;     // Parent-link for failure propagation.
    size_t idx;
    size_t gc_alloc_thresh;
    size_t gc_alloc_accum;

    // Wait queue for tasks waiting for this task.
    rust_wait_queue waiting_tasks;
    rust_alarm alarm;

    rust_task(rust_dom *dom,
              rust_task *spawner);
    ~rust_task();

    void start(uintptr_t exit_task_glue,
               uintptr_t spawnee_fn,
               uintptr_t args,
               size_t callsz);
    void grow(size_t n_frame_bytes);
    bool running();
    bool blocked();
    bool blocked_on(rust_cond *cond);
    bool dead();

    void link_gc(gc_alloc *gcm);
    void unlink_gc(gc_alloc *gcm);
    void *malloc(size_t sz, type_desc *td=0);
    void *realloc(void *data, size_t sz, bool gc_mem=false);
    void free(void *p, bool gc_mem=false);

    const char *state_str();
    void transition(ptr_vec<rust_task> *svec, ptr_vec<rust_task> *dvec);

    void block(rust_cond *on);
    void wakeup(rust_cond *from);
    void die();
    void unblock();

    void check_active() { I(dom, dom->curr_task == this); }
    void check_suspended() { I(dom, dom->curr_task != this); }

    // Swap in some glue code to run when we have returned to the
    // task's context (assuming we're the active task).
    void run_after_return(size_t nargs, uintptr_t glue);

    // Swap in some glue code to run when we're next activated
    // (assuming we're the suspended task).
    void run_on_resume(uintptr_t glue);

    // Save callee-saved registers and return to the main loop.
    void yield(size_t nargs);

    // Fail this task (assuming caller-on-stack is different task).
    void kill();

    // Fail self, assuming caller-on-stack is this task.
    void fail(size_t nargs);

    // Run the gc glue on the task stack.
    void gc(size_t nargs);

    // Disconnect from our supervisor.
    void unsupervise();

    // Notify tasks waiting for us that we are about to die.
    void notify_waiting_tasks();

    uintptr_t get_fp();
    uintptr_t get_previous_fp(uintptr_t fp);
    frame_glue_fns *get_frame_glue_fns(uintptr_t fp);
    rust_crate_cache * get_crate_cache(rust_crate const *curr_crate);
};

struct rust_port : public rc_base<rust_port>,
                   public task_owned<rust_port>,
                   public rust_cond {
    rust_task *task;
    size_t unit_sz;
    ptr_vec<rust_token> writers;
    ptr_vec<rust_chan> chans;

    rust_port(rust_task *task, size_t unit_sz);
    ~rust_port();
};

struct rust_token : public rust_cond {
    rust_chan *chan;      // Link back to the channel this token belongs to
    size_t idx;           // Index into port->writers.
    bool submitted;       // Whether token is in a port->writers.

    rust_token(rust_chan *chan);
    ~rust_token();

    bool pending() const;
    void submit();
    void withdraw();
};


struct circ_buf : public dom_owned<circ_buf> {
    static const size_t INIT_CIRC_BUF_UNITS = 8;
    static const size_t MAX_CIRC_BUF_SIZE = 1 << 24;

    rust_dom *dom;
    size_t alloc;
    size_t unit_sz;
    size_t next;
    size_t unread;
    uint8_t *data;

    circ_buf(rust_dom *dom, size_t unit_sz);
    ~circ_buf();

    void transfer(void *dst);
    void push(void *src);
    void shift(void *dst);
};

#include "rust_chan.h"

int
rust_main_loop(rust_dom *dom);

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
