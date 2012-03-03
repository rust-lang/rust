/* Native builtins. */

#include "rust_internal.h"
#include "rust_task_thread.h"
#include "rust_task.h"
#include "rust_util.h"
#include "rust_scheduler.h"
#include "sync/timer.h"
#include "rust_abi.h"

#ifdef __APPLE__
#include <crt_externs.h>
#endif

#if !defined(__WIN32__)
#include <sys/time.h>
#endif

#ifdef __FreeBSD__
extern char **environ;
#endif

extern "C" CDECL rust_str*
last_os_error() {
    rust_task *task = rust_task_thread::get_task();

    LOG(task, task, "last_os_error()");

#if defined(__WIN32__)
    LPTSTR buf;
    DWORD err = GetLastError();
    DWORD res = FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                              FORMAT_MESSAGE_FROM_SYSTEM |
                              FORMAT_MESSAGE_IGNORE_INSERTS,
                              NULL, err,
                              MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                              (LPTSTR) &buf, 0, NULL);
    if (!res) {
        task->fail();
        return NULL;
    }
#elif defined(_GNU_SOURCE)
    char cbuf[BUF_BYTES];
    char *buf = strerror_r(errno, cbuf, sizeof(cbuf));
    if (!buf) {
        task->fail();
        return NULL;
    }
#else
    char buf[BUF_BYTES];
    int err = strerror_r(errno, buf, sizeof(buf));
    if (err) {
        task->fail();
        return NULL;
    }
#endif

    rust_str * st = make_str(task->kernel, buf, strlen(buf),
                             "last_os_error");
#ifdef __WIN32__
    LocalFree((HLOCAL)buf);
#endif
    return st;
}

extern "C" CDECL rust_str *
rust_getcwd() {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, task, "rust_getcwd()");

    char cbuf[BUF_BYTES];

#if defined(__WIN32__)
    if (!_getcwd(cbuf, sizeof(cbuf))) {
#else
        if (!getcwd(cbuf, sizeof(cbuf))) {
#endif
        task->fail();
        return NULL;
    }

    return make_str(task->kernel, cbuf, strlen(cbuf), "rust_str(getcwd");
}

#if defined(__WIN32__)
extern "C" CDECL rust_vec *
rust_env_pairs() {
    rust_task *task = rust_task_thread::get_task();
    size_t envc = 0;
    LPTCH ch = GetEnvironmentStringsA();
    LPTCH c;
    for (c = ch; *c; c += strlen(c) + 1) {
        ++envc;
    }
    c = ch;
    rust_vec *v = (rust_vec *)
        task->kernel->malloc(vec_size<rust_vec*>(envc),
                       "str vec interior");
    v->fill = v->alloc = sizeof(rust_vec*) * envc;
    for (size_t i = 0; i < envc; ++i) {
        size_t n = strlen(c);
        rust_str *str = make_str(task->kernel, c, n, "str");
        ((rust_str**)&v->data)[i] = str;
        c += n + 1;
    }
    if (ch) {
        FreeEnvironmentStrings(ch);
    }
    return v;
}
#else
extern "C" CDECL rust_vec *
rust_env_pairs() {
    rust_task *task = rust_task_thread::get_task();
#ifdef __APPLE__
    char **environ = *_NSGetEnviron();
#endif
    char **e = environ;
    size_t envc = 0;
    while (*e) {
        ++envc; ++e;
    }
    return make_str_vec(task->kernel, envc, environ);
}
#endif

extern "C" CDECL intptr_t
refcount(intptr_t *v) {
    // Passed-in value has refcount 1 too high
    // because it was ref'ed while making the call.
    return (*v) - 1;
}

extern "C" CDECL void
unsupervise() {
    rust_task *task = rust_task_thread::get_task();
    task->unsupervise();
}

extern "C" CDECL void
vec_reserve_shared(type_desc* ty, rust_vec** vp,
                   size_t n_elts) {
    rust_task *task = rust_task_thread::get_task();
    reserve_vec_exact(task, vp, n_elts * ty->size);
}

extern "C" CDECL void
str_reserve_shared(rust_vec** sp,
                   size_t n_elts) {
    rust_task *task = rust_task_thread::get_task();
    reserve_vec_exact(task, sp, n_elts + 1);
}

/**
 * Copies elements in an unsafe buffer to the given interior vector. The
 * vector must have size zero.
 */
extern "C" CDECL rust_vec*
vec_from_buf_shared(type_desc *ty, void *ptr, size_t count) {
    rust_task *task = rust_task_thread::get_task();
    size_t fill = ty->size * count;
    rust_vec* v = (rust_vec*)task->kernel->malloc(fill + sizeof(rust_vec),
                                                    "vec_from_buf");
    v->fill = v->alloc = fill;
    memmove(&v->data[0], ptr, fill);
    return v;
}

extern "C" CDECL void
rust_str_push(rust_vec** sp, uint8_t byte) {
    rust_task *task = rust_task_thread::get_task();
    size_t fill = (*sp)->fill;
    reserve_vec(task, sp, fill + 1);
    (*sp)->data[fill-1] = byte;
    (*sp)->data[fill] = 0;
    (*sp)->fill = fill + 1;
}

extern "C" CDECL void *
rand_new() {
    rust_task *task = rust_task_thread::get_task();
    rust_task_thread *thread = task->thread;
    randctx *rctx = (randctx *) task->malloc(sizeof(randctx), "randctx");
    if (!rctx) {
        task->fail();
        return NULL;
    }
    isaac_init(thread->kernel, rctx);
    return rctx;
}

extern "C" CDECL size_t
rand_next(randctx *rctx) {
    return isaac_rand(rctx);
}

extern "C" CDECL void
rand_free(randctx *rctx) {
    rust_task *task = rust_task_thread::get_task();
    task->free(rctx);
}

/* Debug builtins for std::dbg. */

static void
debug_tydesc_helper(type_desc *t)
{
    rust_task *task = rust_task_thread::get_task();
    LOG(task, stdlib, "  size %" PRIdPTR ", align %" PRIdPTR
        ", first_param 0x%" PRIxPTR,
        t->size, t->align, t->first_param);
}

extern "C" CDECL void
debug_tydesc(type_desc *t) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, stdlib, "debug_tydesc");
    debug_tydesc_helper(t);
}

extern "C" CDECL void
debug_opaque(type_desc *t, uint8_t *front) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, stdlib, "debug_opaque");
    debug_tydesc_helper(t);
    // FIXME may want to actually account for alignment.  `front` may not
    // indeed be the front byte of the passed-in argument.
    for (uintptr_t i = 0; i < t->size; ++front, ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, *front);
    }
}

// FIXME this no longer reflects the actual structure of boxes!
struct rust_box {
    RUST_REFCOUNTED(rust_box)

    // FIXME `data` could be aligned differently from the actual box body data
    uint8_t data[];
};

extern "C" CDECL void
debug_box(type_desc *t, rust_box *box) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, stdlib, "debug_box(0x%" PRIxPTR ")", box);
    debug_tydesc_helper(t);
    LOG(task, stdlib, "  refcount %" PRIdPTR,
        box->ref_count - 1);  // -1 because we ref'ed for this call
    for (uintptr_t i = 0; i < t->size; ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, box->data[i]);
    }
}

struct rust_tag {
    uintptr_t discriminant;
    uint8_t variant[];
};

extern "C" CDECL void
debug_tag(type_desc *t, rust_tag *tag) {
    rust_task *task = rust_task_thread::get_task();

    LOG(task, stdlib, "debug_tag");
    debug_tydesc_helper(t);
    LOG(task, stdlib, "  discriminant %" PRIdPTR, tag->discriminant);

    for (uintptr_t i = 0; i < t->size - sizeof(tag->discriminant); ++i)
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i,
            tag->variant[i]);
}

struct rust_fn {
    uintptr_t *thunk;
    rust_box *closure;
};

extern "C" CDECL void
debug_fn(type_desc *t, rust_fn *fn) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, stdlib, "debug_fn");
    debug_tydesc_helper(t);
    LOG(task, stdlib, "  thunk at 0x%" PRIxPTR, fn->thunk);
    LOG(task, stdlib, "  closure at 0x%" PRIxPTR, fn->closure);
    if (fn->closure) {
        LOG(task, stdlib, "    refcount %" PRIdPTR, fn->closure->ref_count);
    }
}

extern "C" CDECL void *
debug_ptrcast(type_desc *from_ty,
              type_desc *to_ty,
              void *ptr) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, stdlib, "debug_ptrcast from");
    debug_tydesc_helper(from_ty);
    LOG(task, stdlib, "to");
    debug_tydesc_helper(to_ty);
    return ptr;
}

extern "C" CDECL void *
debug_get_stk_seg() {
    rust_task *task = rust_task_thread::get_task();
    return task->stk;
}

extern "C" CDECL rust_vec*
rust_list_files(rust_str *path) {
    rust_task *task = rust_task_thread::get_task();
    array_list<rust_str*> strings;
#if defined(__WIN32__)
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile((char*)path->data, &FindFileData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            rust_str *str = make_str(task->kernel, FindFileData.cFileName,
                                     strlen(FindFileData.cFileName),
                                     "list_files_str");
            strings.push(str);
        } while (FindNextFile(hFind, &FindFileData));
        FindClose(hFind);
    }
#else
    DIR *dirp = opendir((char*)path->data);
  if (dirp) {
      struct dirent *dp;
      while ((dp = readdir(dirp))) {
          rust_vec *str = make_str(task->kernel, dp->d_name,
                                    strlen(dp->d_name),
                                    "list_files_str");
          strings.push(str);
      }
      closedir(dirp);
  }
#endif

  rust_vec *vec = (rust_vec *)
      task->kernel->malloc(vec_size<rust_vec*>(strings.size()),
                           "list_files_vec");
  size_t alloc_sz = sizeof(rust_vec*) * strings.size();
  vec->fill = vec->alloc = alloc_sz;
  memcpy(&vec->data[0], strings.data(), alloc_sz);
  return vec;
}

extern "C" CDECL int
rust_path_is_dir(char *path) {
    struct stat buf;
    if (stat(path, &buf)) {
        return 0;
    }
    return S_ISDIR(buf.st_mode);
}

extern "C" CDECL int
rust_path_exists(char *path) {
    struct stat buf;
    if (stat(path, &buf)) {
        return 0;
    }
    return 1;
}

extern "C" CDECL FILE* rust_get_stdin() {return stdin;}
extern "C" CDECL FILE* rust_get_stdout() {return stdout;}
extern "C" CDECL FILE* rust_get_stderr() {return stderr;}

extern "C" CDECL int
rust_ptr_eq(type_desc *t, rust_box *a, rust_box *b) {
    return a == b;
}

#if defined(__WIN32__)
extern "C" CDECL void
get_time(uint32_t *sec, uint32_t *usec) {
    FILETIME fileTime;
    GetSystemTimeAsFileTime(&fileTime);

    // A FILETIME contains a 64-bit value representing the number of
    // hectonanosecond (100-nanosecond) intervals since 1601-01-01T00:00:00Z.
    // http://support.microsoft.com/kb/167296/en-us
    ULARGE_INTEGER ul;
    ul.LowPart = fileTime.dwLowDateTime;
    ul.HighPart = fileTime.dwHighDateTime;
    uint64_t ns_since_1601 = ul.QuadPart / 10;

    const uint64_t NANOSECONDS_FROM_1601_TO_1970 = 11644473600000000u;
    uint64_t ns_since_1970 = ns_since_1601 - NANOSECONDS_FROM_1601_TO_1970;
    *sec = ns_since_1970 / 1000000;
    *usec = ns_since_1970 % 1000000;
}
#else
extern "C" CDECL void
get_time(uint32_t *sec, uint32_t *usec) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *sec = tv.tv_sec;
    *usec = tv.tv_usec;
}
#endif

extern "C" CDECL void
precise_time_ns(uint64_t *ns) {
    timer t;
    *ns = t.time_ns();
}

extern "C" CDECL rust_sched_id
rust_get_sched_id() {
    rust_task *task = rust_task_thread::get_task();
    return task->sched->get_id();
}

extern "C" CDECL rust_sched_id
rust_new_sched(uintptr_t threads) {
    rust_task *task = rust_task_thread::get_task();
    A(task->thread, threads > 0,
      "Can't create a scheduler with no threads, silly!");
    return task->kernel->create_scheduler(threads);
}

extern "C" CDECL rust_task_id
get_task_id() {
    rust_task *task = rust_task_thread::get_task();
    return task->id;
}

static rust_task_id
new_task_common(rust_scheduler *sched, rust_task *parent) {
    return sched->create_task(parent, NULL);
}

extern "C" CDECL rust_task_id
new_task() {
    rust_task *task = rust_task_thread::get_task();
    return new_task_common(task->sched, task);
}

extern "C" CDECL rust_task_id
rust_new_task_in_sched(rust_sched_id id) {
    rust_task *task = rust_task_thread::get_task();
    rust_scheduler *sched = task->kernel->get_scheduler_by_id(id);
    // FIXME: What if we didn't get the scheduler?
    return new_task_common(sched, task);
}

extern "C" CDECL void
rust_task_config_notify(rust_task_id task_id, chan_handle *chan) {
    rust_task *task = rust_task_thread::get_task();
    rust_task *target = task->kernel->get_task_by_id(task_id);
    A(task->thread, target != NULL,
      "This function should only be called when we know the task exists");
    target->config_notify(*chan);
    target->deref();
}

extern "C" rust_task *
rust_get_task() {
    return rust_task_thread::get_task();
}

extern "C" CDECL void
start_task(rust_task_id id, fn_env_pair *f) {
    rust_task *task = rust_task_thread::get_task();
    rust_task *target = task->kernel->get_task_by_id(id);
    target->start(f->f, f->env, NULL);
    target->deref();
}

extern "C" CDECL int
sched_threads() {
    rust_task *task = rust_task_thread::get_task();
    return task->sched->number_of_threads();
}

extern "C" CDECL rust_port*
new_port(size_t unit_sz) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, comm, "new_port(task=0x%" PRIxPTR " (%s), unit_sz=%d)",
        (uintptr_t) task, task->name, unit_sz);
    // port starts with refcount == 1
    return new (task->kernel, "rust_port") rust_port(task, unit_sz);
}

extern "C" CDECL void
rust_port_detach(rust_port *port) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, comm, "rust_port_detach(0x%" PRIxPTR ")", (uintptr_t) port);
    port->detach();
}

extern "C" CDECL void
del_port(rust_port *port) {
    rust_task *task = rust_task_thread::get_task();
    LOG(task, comm, "del_port(0x%" PRIxPTR ")", (uintptr_t) port);
    A(task->thread, port->ref_count == 1, "Expected port ref_count == 1");
    port->deref();
}

extern "C" CDECL size_t
rust_port_size(rust_port *port) {
    return port->size();
}

extern "C" CDECL rust_port_id
get_port_id(rust_port *port) {
    return port->id;
}

extern "C" CDECL uintptr_t
chan_id_send(type_desc *t, rust_task_id target_task_id,
             rust_port_id target_port_id, void *sptr) {
    // FIXME: make sure this is thread-safe
    bool sent = false;
    rust_task *task = rust_task_thread::get_task();

    LOG(task, comm, "chan_id_send task: 0x%" PRIxPTR
        " port: 0x%" PRIxPTR, (uintptr_t) target_task_id,
        (uintptr_t) target_port_id);

    rust_task *target_task = task->kernel->get_task_by_id(target_task_id);
    if(target_task) {
        rust_port *port = target_task->get_port_by_id(target_port_id);
        if(port) {
            port->send(sptr);
            scoped_lock with(target_task->port_lock);
            port->deref();
            sent = true;
        } else {
            LOG(task, comm, "didn't get the port");
        }
        target_task->deref();
    } else {
        LOG(task, comm, "didn't get the task");
    }
    return (uintptr_t)sent;
}

// This is called by an intrinsic on the Rust stack and must run
// entirely in the red zone. Do not call on the C stack.
extern "C" CDECL void
rust_task_yield(rust_task *task, bool *killed) {
    task->yield(killed);
}

extern "C" CDECL void
port_recv(uintptr_t *dptr, rust_port *port,
          uintptr_t *yield, uintptr_t *killed) {
    *yield = false;
    *killed = false;
    rust_task *task = rust_task_thread::get_task();
    {
        scoped_lock with(port->lock);

        LOG(task, comm, "port: 0x%" PRIxPTR ", dptr: 0x%" PRIxPTR
            ", size: 0x%" PRIxPTR,
            (uintptr_t) port, (uintptr_t) dptr, port->unit_sz);

        if (port->receive(dptr)) {
            return;
        }

        // If this task has been killed then we're not going to bother
        // blocking, we have to unwind.
        if (task->must_fail_from_being_killed()) {
            *killed = true;
            return;
        }

        // No data was buffered on any incoming channel, so block this task on
        // the port. Remember the rendezvous location so that any sender task
        // can write to it before waking up this task.

        LOG(task, comm, "<=== waiting for rendezvous data ===");
        task->rendezvous_ptr = dptr;
        task->block(port, "waiting for rendezvous data");
    }
    *yield = true;
    return;
}

extern "C" CDECL void
rust_port_select(rust_port **dptr, rust_port **ports,
                 size_t n_ports, uintptr_t *yield) {
    rust_task *task = rust_task_thread::get_task();
    rust_port_selector *selector = task->get_port_selector();
    selector->select(task, dptr, ports, n_ports, yield);
}

extern "C" CDECL void
rust_set_exit_status(intptr_t code) {
    rust_task *task = rust_task_thread::get_task();
    task->kernel->set_exit_status((int)code);
}

extern void log_console_on();

extern "C" CDECL void
rust_log_console_on() {
    log_console_on();
}

extern void log_console_off(rust_env *env);

extern "C" CDECL void
rust_log_console_off() {
    rust_task *task = rust_task_thread::get_task();
    log_console_off(task->kernel->env);
}

extern "C" CDECL lock_and_signal *
rust_dbg_lock_create() {
    return new lock_and_signal();
}

extern "C" CDECL void
rust_dbg_lock_destroy(lock_and_signal *lock) {
    rust_task *task = rust_task_thread::get_task();
    I(task->thread, lock);
    delete lock;
}

extern "C" CDECL void
rust_dbg_lock_lock(lock_and_signal *lock) {
    rust_task *task = rust_task_thread::get_task();
    I(task->thread, lock);
    lock->lock();
}

extern "C" CDECL void
rust_dbg_lock_unlock(lock_and_signal *lock) {
    rust_task *task = rust_task_thread::get_task();
    I(task->thread, lock);
    lock->unlock();
}

extern "C" CDECL void
rust_dbg_lock_wait(lock_and_signal *lock) {
    rust_task *task = rust_task_thread::get_task();
    I(task->thread, lock);
    lock->wait();
}

extern "C" CDECL void
rust_dbg_lock_signal(lock_and_signal *lock) {
    rust_task *task = rust_task_thread::get_task();
    I(task->thread, lock);
    lock->signal();
}

typedef void *(*dbg_callback)(void*);

extern "C" CDECL void *
rust_dbg_call(dbg_callback cb, void *data) {
    return cb(data);
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
