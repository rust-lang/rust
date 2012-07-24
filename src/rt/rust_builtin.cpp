/* Foreign builtins. */

#include "rust_sched_loop.h"
#include "rust_task.h"
#include "rust_util.h"
#include "rust_scheduler.h"
#include "sync/timer.h"
#include "rust_abi.h"
#include "rust_port.h"
#include "rust_cond_lock.h"

#include <time.h>

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
    rust_task *task = rust_get_current_task();

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
    rust_task *task = rust_get_current_task();
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

    return make_str(task->kernel, cbuf, strlen(cbuf), "rust_str(getcwd)");
}

#if defined(__WIN32__)
extern "C" CDECL rust_vec_box *
rust_env_pairs() {
    rust_task *task = rust_get_current_task();
    size_t envc = 0;
    LPTCH ch = GetEnvironmentStringsA();
    LPTCH c;
    for (c = ch; *c; c += strlen(c) + 1) {
        ++envc;
    }
    c = ch;
    rust_vec_box *v = (rust_vec_box *)
        task->kernel->malloc(vec_size<rust_vec_box*>(envc),
                       "str vec interior");
    v->body.fill = v->body.alloc = sizeof(rust_vec*) * envc;
    for (size_t i = 0; i < envc; ++i) {
        size_t n = strlen(c);
        rust_str *str = make_str(task->kernel, c, n, "str");
        ((rust_str**)&v->body.data)[i] = str;
        c += n + 1;
    }
    if (ch) {
        FreeEnvironmentStrings(ch);
    }
    return v;
}
#else
extern "C" CDECL rust_vec_box *
rust_env_pairs() {
    rust_task *task = rust_get_current_task();
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

extern "C" CDECL void
vec_reserve_shared_actual(type_desc* ty, rust_vec_box** vp,
                          size_t n_elts) {
    rust_task *task = rust_get_current_task();
    reserve_vec_exact_shared(task, vp, n_elts * ty->size);
}

// This is completely misnamed.
extern "C" CDECL void
vec_reserve_shared(type_desc* ty, rust_vec_box** vp,
                   size_t n_elts) {
    rust_task *task = rust_get_current_task();
    reserve_vec_exact(task, vp, n_elts * ty->size);
}

extern "C" CDECL void
str_reserve_shared(rust_vec_box** sp,
                   size_t n_elts) {
    rust_task *task = rust_get_current_task();
    reserve_vec_exact(task, sp, n_elts + 1);
}

/**
 * Copies elements in an unsafe buffer to the given interior vector. The
 * vector must have size zero.
 */
extern "C" CDECL rust_vec_box*
vec_from_buf_shared(type_desc *ty, void *ptr, size_t count) {
    rust_task *task = rust_get_current_task();
    size_t fill = ty->size * count;
    rust_vec_box* v = (rust_vec_box*)
        task->kernel->malloc(fill + sizeof(rust_vec_box),
                             "vec_from_buf");
    v->body.fill = v->body.alloc = fill;
    memmove(&v->body.data[0], ptr, fill);
    return v;
}

extern "C" CDECL void
rust_str_push(rust_vec_box** sp, uint8_t byte) {
    rust_task *task = rust_get_current_task();
    size_t fill = (*sp)->body.fill;
    reserve_vec(task, sp, fill + 1);
    (*sp)->body.data[fill-1] = byte;
    (*sp)->body.data[fill] = 0;
    (*sp)->body.fill = fill + 1;
}

extern "C" CDECL rust_vec*
rand_seed() {
    size_t size = sizeof(ub4) * RANDSIZ;
    rust_task *task = rust_get_current_task();
    rust_vec *v = (rust_vec *) task->kernel->malloc(vec_size<uint8_t>(size),
                                            "rand_seed");
    v->fill = v->alloc = size;
    isaac_seed(task->kernel, (uint8_t*) &v->data);
    return v;
}

extern "C" CDECL void *
rand_new() {
    rust_task *task = rust_get_current_task();
    rust_sched_loop *thread = task->sched_loop;
    randctx *rctx = (randctx *) task->malloc(sizeof(randctx), "rand_new");
    if (!rctx) {
        task->fail();
        return NULL;
    }
    isaac_init(thread->kernel, rctx, NULL);
    return rctx;
}

extern "C" CDECL void *
rand_new_seeded(rust_vec_box* seed) {
    rust_task *task = rust_get_current_task();
    rust_sched_loop *thread = task->sched_loop;
    randctx *rctx = (randctx *) task->malloc(sizeof(randctx),
                                             "rand_new_seeded");
    if (!rctx) {
        task->fail();
        return NULL;
    }
    isaac_init(thread->kernel, rctx, seed);
    return rctx;
}

extern "C" CDECL size_t
rand_next(randctx *rctx) {
    return isaac_rand(rctx);
}

extern "C" CDECL void
rand_free(randctx *rctx) {
    rust_task *task = rust_get_current_task();
    task->free(rctx);
}


/* Debug helpers strictly to verify ABI conformance.
 *
 * FIXME (#2665): move these into a testcase when the testsuite
 * understands how to have explicit C files included.
 */

struct quad {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
};

struct floats {
    double a;
    uint8_t b;
    double c;
};

extern "C" quad
debug_abi_1(quad q) {
    quad qq = { q.c + 1,
                q.d - 1,
                q.a + 1,
                q.b - 1 };
    return qq;
}

extern "C" floats
debug_abi_2(floats f) {
    floats ff = { f.c + 1.0,
                  0xff,
                  f.a - 1.0 };
    return ff;
}

/* Debug builtins for std::dbg. */

static void
debug_tydesc_helper(type_desc *t)
{
    rust_task *task = rust_get_current_task();
    LOG(task, stdlib, "  size %" PRIdPTR ", align %" PRIdPTR,
        t->size, t->align);
}

extern "C" CDECL void
debug_tydesc(type_desc *t) {
    rust_task *task = rust_get_current_task();
    LOG(task, stdlib, "debug_tydesc");
    debug_tydesc_helper(t);
}

extern "C" CDECL void
debug_opaque(type_desc *t, uint8_t *front) {
    rust_task *task = rust_get_current_task();
    LOG(task, stdlib, "debug_opaque");
    debug_tydesc_helper(t);
    // FIXME (#2667) may want to actually account for alignment.
    // `front` may not indeed be the front byte of the passed-in
    // argument.
    for (uintptr_t i = 0; i < t->size; ++front, ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, *front);
    }
}

// FIXME (#2667) this no longer reflects the actual structure of boxes!
struct rust_box {
    RUST_REFCOUNTED(rust_box)

    // FIXME (#2667) `data` could be aligned differently from the actual
    // box body data
    uint8_t data[];
};

extern "C" CDECL void
debug_box(type_desc *t, rust_box *box) {
    rust_task *task = rust_get_current_task();
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
    rust_task *task = rust_get_current_task();

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
    rust_task *task = rust_get_current_task();
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
    rust_task *task = rust_get_current_task();
    LOG(task, stdlib, "debug_ptrcast from");
    debug_tydesc_helper(from_ty);
    LOG(task, stdlib, "to");
    debug_tydesc_helper(to_ty);
    return ptr;
}

extern "C" CDECL void *
debug_get_stk_seg() {
    rust_task *task = rust_get_current_task();
    return task->stk;
}

extern "C" CDECL rust_vec_box*
rust_list_files(rust_str *path) {
    rust_task *task = rust_get_current_task();
    array_list<rust_str*> strings;
#if defined(__WIN32__)
    WIN32_FIND_DATA FindFileData;
    HANDLE hFind = FindFirstFile((char*)path->body.data, &FindFileData);
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
    DIR *dirp = opendir((char*)path->body.data);
  if (dirp) {
      struct dirent *dp;
      while ((dp = readdir(dirp))) {
          rust_vec_box *str = make_str(task->kernel, dp->d_name,
                                       strlen(dp->d_name),
                                       "list_files_str");
          strings.push(str);
      }
      closedir(dirp);
  }
#endif

  rust_vec_box *vec = (rust_vec_box *)
      task->kernel->malloc(vec_size<rust_vec_box*>(strings.size()),
                           "list_files_vec");
  size_t alloc_sz = sizeof(rust_vec*) * strings.size();
  vec->body.fill = vec->body.alloc = alloc_sz;
  memcpy(&vec->body.data[0], strings.data(), alloc_sz);
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
get_time(int64_t *sec, int32_t *nsec) {
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
    *nsec = (ns_since_1970 % 1000000) * 1000;
}
#else
extern "C" CDECL void
get_time(int64_t *sec, int32_t *nsec) {
#ifdef __APPLE__
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *sec = tv.tv_sec;
    *nsec = tv.tv_usec * 1000;
#else
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *sec = ts.tv_sec;
    *nsec = ts.tv_nsec;
#endif
}
#endif

extern "C" CDECL void
precise_time_ns(uint64_t *ns) {
    timer t;
    *ns = t.time_ns();
}

struct rust_tm {
    int32_t tm_sec;
    int32_t tm_min;
    int32_t tm_hour;
    int32_t tm_mday;
    int32_t tm_mon;
    int32_t tm_year;
    int32_t tm_wday;
    int32_t tm_yday;
    int32_t tm_isdst;
    int32_t tm_gmtoff;
    rust_str *tm_zone;
    int32_t tm_nsec;
};

void rust_tm_to_tm(rust_tm* in_tm, tm* out_tm) {
    memset(out_tm, 0, sizeof(tm));
    out_tm->tm_sec = in_tm->tm_sec;
    out_tm->tm_min = in_tm->tm_min;
    out_tm->tm_hour = in_tm->tm_hour;
    out_tm->tm_mday = in_tm->tm_mday;
    out_tm->tm_mon = in_tm->tm_mon;
    out_tm->tm_year = in_tm->tm_year;
    out_tm->tm_wday = in_tm->tm_wday;
    out_tm->tm_yday = in_tm->tm_yday;
    out_tm->tm_isdst = in_tm->tm_isdst;
}

void tm_to_rust_tm(tm* in_tm, rust_tm* out_tm, int32_t gmtoff,
                   const char *zone, int32_t nsec) {
    out_tm->tm_sec = in_tm->tm_sec;
    out_tm->tm_min = in_tm->tm_min;
    out_tm->tm_hour = in_tm->tm_hour;
    out_tm->tm_mday = in_tm->tm_mday;
    out_tm->tm_mon = in_tm->tm_mon;
    out_tm->tm_year = in_tm->tm_year;
    out_tm->tm_wday = in_tm->tm_wday;
    out_tm->tm_yday = in_tm->tm_yday;
    out_tm->tm_isdst = in_tm->tm_isdst;
    out_tm->tm_gmtoff = gmtoff;
    out_tm->tm_nsec = nsec;

    if (zone != NULL) {
        size_t size = strlen(zone);
        str_reserve_shared(&out_tm->tm_zone, size);
        memcpy(out_tm->tm_zone->body.data, zone, size);
        out_tm->tm_zone->body.fill = size + 1;
        out_tm->tm_zone->body.data[size] = '\0';
    }
}

#if defined(__WIN32__)
#define TZSET() _tzset()
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
#define GMTIME(clock, result) gmtime_s((result), (clock))
#define LOCALTIME(clock, result) localtime_s((result), (clock))
#define TIMEGM(result) _mkgmtime64(result)
#else
struct tm* GMTIME(const time_t *clock, tm *result) {
    struct tm* t = gmtime(clock);
    if (t == NULL || result == NULL) { return NULL; }
    *result = *t;
    return result;
}
struct tm* LOCALTIME(const time_t *clock, tm *result) {
    struct tm* t = localtime(clock);
    if (t == NULL || result == NULL) { return NULL; }
    *result = *t;
    return result;
}
#define TIMEGM(result) mktime((result)) - _timezone
#endif
#else
#define TZSET() tzset()
#define GMTIME(clock, result) gmtime_r((clock), (result))
#define LOCALTIME(clock, result) localtime_r((clock), (result))
#define TIMEGM(result) timegm(result)
#endif

extern "C" CDECL void
rust_tzset() {
    TZSET();
}

extern "C" CDECL void
rust_gmtime(int64_t *sec, int32_t *nsec, rust_tm *timeptr) {
    tm tm;
    time_t s = *sec;
    GMTIME(&s, &tm);

    tm_to_rust_tm(&tm, timeptr, 0, "UTC", *nsec);
}

extern "C" CDECL void
rust_localtime(int64_t *sec, int32_t *nsec, rust_tm *timeptr) {
    tm tm;
    time_t s = *sec;
    LOCALTIME(&s, &tm);

#if defined(__WIN32__)
    int32_t gmtoff = -timezone;
    char zone[64];
    strftime(zone, sizeof(zone), "%Z", &tm);
#else
    int32_t gmtoff = tm.tm_gmtoff;
    const char *zone = tm.tm_zone;
#endif

    tm_to_rust_tm(&tm, timeptr, gmtoff, zone, *nsec);
}

extern "C" CDECL void
rust_timegm(rust_tm* timeptr, int64_t *out) {
    tm t;
    rust_tm_to_tm(timeptr, &t);
    *out = TIMEGM(&t);
}

extern "C" CDECL void
rust_mktime(rust_tm* timeptr, int64_t *out) {
    tm t;
    rust_tm_to_tm(timeptr, &t);
    *out = mktime(&t);
}

extern "C" CDECL rust_sched_id
rust_get_sched_id() {
    rust_task *task = rust_get_current_task();
    return task->sched->get_id();
}

extern "C" CDECL rust_sched_id
rust_new_sched(uintptr_t threads) {
    rust_task *task = rust_get_current_task();
    assert(threads > 0 && "Can't create a scheduler with no threads, silly!");
    return task->kernel->create_scheduler(threads);
}

extern "C" CDECL rust_task_id
get_task_id() {
    rust_task *task = rust_get_current_task();
    return task->id;
}

static rust_task*
new_task_common(rust_scheduler *sched, rust_task *parent) {
    return sched->create_task(parent, NULL);
}

extern "C" CDECL rust_task*
new_task() {
    rust_task *task = rust_get_current_task();
    return new_task_common(task->sched, task);
}

extern "C" CDECL rust_task*
rust_new_task_in_sched(rust_sched_id id) {
    rust_task *task = rust_get_current_task();
    rust_scheduler *sched = task->kernel->get_scheduler_by_id(id);
    if (sched == NULL)
        return NULL;
    return new_task_common(sched, task);
}

extern "C" CDECL void
rust_task_config_notify(rust_task *target, rust_port_id *port) {
    target->config_notify(*port);
}

extern "C" rust_task *
rust_get_task() {
    return rust_get_current_task();
}

extern "C" CDECL void
start_task(rust_task *target, fn_env_pair *f) {
    target->start(f->f, f->env, NULL);
}

extern "C" CDECL int
sched_threads() {
    rust_task *task = rust_get_current_task();
    return task->sched->number_of_threads();
}

extern "C" CDECL rust_port*
rust_port_take(rust_port_id id) {
    rust_task *task = rust_get_current_task();
    return task->kernel->get_port_by_id(id);
}

extern "C" CDECL void
rust_port_drop(rust_port *p) {
    assert(p != NULL);
    p->deref();
}

extern "C" CDECL rust_task_id
rust_port_task(rust_port *p) {
    assert(p != NULL);
    return p->task->id;
}

extern "C" CDECL rust_port*
new_port(size_t unit_sz) {
    rust_task *task = rust_get_current_task();
    LOG(task, comm, "new_port(task=0x%" PRIxPTR " (%s), unit_sz=%d)",
        (uintptr_t) task, task->name, unit_sz);
    // port starts with refcount == 1
    return new (task->kernel, "rust_port") rust_port(task, unit_sz);
}

extern "C" CDECL void
rust_port_begin_detach(rust_port *port, uintptr_t *yield) {
    rust_task *task = rust_get_current_task();
    LOG(task, comm, "rust_port_detach(0x%" PRIxPTR ")", (uintptr_t) port);
    port->begin_detach(yield);
}

extern "C" CDECL void
rust_port_end_detach(rust_port *port) {
    port->end_detach();
}

extern "C" CDECL void
del_port(rust_port *port) {
    rust_task *task = rust_get_current_task();
    LOG(task, comm, "del_port(0x%" PRIxPTR ")", (uintptr_t) port);
    delete port;
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
rust_port_id_send(rust_port_id target_port_id, void *sptr) {
    rust_task *task = rust_get_current_task();
    return (uintptr_t)task->kernel->send_to_port(target_port_id, sptr);
}

// This is called by an intrinsic on the Rust stack and must run
// entirely in the red zone. Do not call on the C stack.
extern "C" CDECL MUST_CHECK bool
rust_task_yield(rust_task *task, bool *killed) {
    return task->yield();
}

extern "C" CDECL void
port_recv(uintptr_t *dptr, rust_port *port, uintptr_t *yield) {
    port->receive(dptr, yield);
}

extern "C" CDECL void
rust_port_select(rust_port **dptr, rust_port **ports,
                 size_t n_ports, uintptr_t *yield) {
    rust_task *task = rust_get_current_task();
    rust_port_selector *selector = task->get_port_selector();
    selector->select(task, dptr, ports, n_ports, yield);
}

extern "C" CDECL void
rust_set_exit_status(intptr_t code) {
    rust_task *task = rust_get_current_task();
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
    rust_task *task = rust_get_current_task();
    log_console_off(task->kernel->env);
}

extern "C" CDECL lock_and_signal *
rust_dbg_lock_create() {
    return new lock_and_signal();
}

extern "C" CDECL void
rust_dbg_lock_destroy(lock_and_signal *lock) {
    assert(lock);
    delete lock;
}

extern "C" CDECL void
rust_dbg_lock_lock(lock_and_signal *lock) {
    assert(lock);
    lock->lock();
}

extern "C" CDECL void
rust_dbg_lock_unlock(lock_and_signal *lock) {
    assert(lock);
    lock->unlock();
}

extern "C" CDECL void
rust_dbg_lock_wait(lock_and_signal *lock) {
    assert(lock);
    lock->wait();
}

extern "C" CDECL void
rust_dbg_lock_signal(lock_and_signal *lock) {
    assert(lock);
    lock->signal();
}

typedef void *(*dbg_callback)(void*);

extern "C" CDECL void *
rust_dbg_call(dbg_callback cb, void *data) {
    return cb(data);
}

extern "C" CDECL void
rust_dbg_breakpoint() {
    BREAKPOINT_AWESOME;
}

extern "C" CDECL rust_sched_id
rust_osmain_sched_id() {
    rust_task *task = rust_get_current_task();
    return task->kernel->osmain_sched_id();
}

extern "C" CDECL bool
rust_compare_and_swap_ptr(intptr_t *address,
                          intptr_t oldval, intptr_t newval) {
    return sync::compare_and_swap(address, oldval, newval);
}

extern "C" CDECL intptr_t
rust_atomic_increment(intptr_t *address) {
    return sync::increment(address);
}

extern "C" CDECL intptr_t
rust_atomic_decrement(intptr_t *address) {
    return sync::decrement(address);
}

extern "C" CDECL void
rust_task_weaken(rust_port_id chan) {
    rust_task *task = rust_get_current_task();
    task->kernel->weaken_task(chan);
}

extern "C" CDECL void
rust_task_unweaken(rust_port_id chan) {
    rust_task *task = rust_get_current_task();
    task->kernel->unweaken_task(chan);
}

extern "C" CDECL uintptr_t*
rust_global_env_chan_ptr() {
    rust_task *task = rust_get_current_task();
    return task->kernel->get_global_env_chan();
}

extern "C" void
rust_task_inhibit_kill(rust_task *task) {
    task->inhibit_kill();
}

extern "C" void
rust_task_allow_kill(rust_task *task) {
    task->allow_kill();
}

extern "C" void
rust_task_inhibit_yield(rust_task *task) {
    task->inhibit_yield();
}

extern "C" void
rust_task_allow_yield(rust_task *task) {
    task->allow_yield();
}

extern "C" void
rust_task_kill_other(rust_task *task) { /* Used for linked failure */
    task->kill();
}

extern "C" void
rust_task_kill_all(rust_task *task) { /* Used for linked failure */
    task->fail_sched_loop();
    // This must not happen twice.
    static bool main_taskgroup_failed = false;
    assert(!main_taskgroup_failed);
    main_taskgroup_failed = true;
}

extern "C" CDECL
bool rust_task_is_unwinding(rust_task *rt) {
    return rt->unwinding;
}

extern "C" rust_cond_lock*
rust_create_cond_lock() {
    return new rust_cond_lock();
}

extern "C" void
rust_destroy_cond_lock(rust_cond_lock *lock) {
    delete lock;
}

extern "C" void
rust_lock_cond_lock(rust_cond_lock *lock) {
    lock->lock.lock();
}

extern "C" void
rust_unlock_cond_lock(rust_cond_lock *lock) {
    lock->lock.unlock();
}

// The next two functions do not use the built in condition variable features
// because the Rust schedule is not aware of them, and they can block the
// scheduler thread.

extern "C" void
rust_wait_cond_lock(rust_cond_lock *lock) {
    assert(false && "condition->wait() is totally broken! Don't use it!");
    rust_task *task = rust_get_current_task();
    lock->lock.must_have_lock();
    assert(NULL == lock->waiting);
    lock->waiting = task;
    task->block(lock, "waiting for signal");
    lock->lock.unlock();
    bool killed = task->yield();
    assert(!killed && "unimplemented");
    lock->lock.lock();
}

extern "C" bool
rust_signal_cond_lock(rust_cond_lock *lock) {
    assert(false && "condition->signal() is totally broken! Don't use it!");
    lock->lock.must_have_lock();
    if(NULL == lock->waiting) {
        return false;
    }
    else {
        lock->waiting->wakeup(lock);
        lock->waiting = NULL;
        return true;
    }
}

// set/get/atexit task_local_data can run on the rust stack for speed.
extern "C" void *
rust_get_task_local_data(rust_task *task) {
    return task->task_local_data;
}
extern "C" void
rust_set_task_local_data(rust_task *task, void *data) {
    task->task_local_data = data;
}
extern "C" void
rust_task_local_data_atexit(rust_task *task, void (*cleanup_fn)(void *data)) {
    task->task_local_data_cleanup = cleanup_fn;
}

extern "C" void
task_clear_event_reject(rust_task *task) {
    task->clear_event_reject();
}

// Waits on an event, returning the pointer to the event that unblocked this
// task.
extern "C" MUST_CHECK bool
task_wait_event(rust_task *task, void **result) {
    // Maybe (if not too slow) assert that the passed in task is the currently
    // running task. We wouldn't want to wait some other task.

    return task->wait_event(result);
}

extern "C" void
task_signal_event(rust_task *target, void *event) {
    target->signal_event(event);
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
