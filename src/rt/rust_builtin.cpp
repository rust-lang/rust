// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* Foreign builtins. */

#include "rust_sched_loop.h"
#include "rust_task.h"
#include "rust_util.h"
#include "rust_scheduler.h"
#include "sync/timer.h"
#include "sync/rust_thread.h"
#include "rust_abi.h"

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

#ifdef __ANDROID__
time_t
timegm(struct tm *tm)
{
    time_t ret;
    char *tz;

    tz = getenv("TZ");
    setenv("TZ", "", 1);
    tzset();
    ret = mktime(tm);
    if (tz)
        setenv("TZ", tz, 1);
    else
        unsetenv("TZ");
    tzset();
    return ret;
}
#endif

#if defined(__WIN32__)
extern "C" CDECL char**
rust_env_pairs() {
    return 0;
}
#else
extern "C" CDECL char**
rust_env_pairs() {
#ifdef __APPLE__
    char **environ = *_NSGetEnviron();
#endif
    return environ;
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
    reserve_vec_exact(vp, n_elts * ty->size);
}

extern "C" CDECL size_t
rand_seed_size() {
    return rng_seed_size();
}

extern "C" CDECL void
rand_gen_seed(uint8_t* dest, size_t size) {
    rng_gen_seed(dest, size);
}

extern "C" CDECL void *
rand_new_seeded(uint8_t* seed, size_t seed_size) {
    rust_task *task = rust_get_current_task();
    rust_rng *rng = (rust_rng *) task->malloc(sizeof(rust_rng),
                                              "rand_new_seeded");
    if (!rng) {
        task->fail();
        return NULL;
    }
    char *env_seed = task->kernel->env->rust_seed;
    rng_init(rng, env_seed, seed, seed_size);
    return rng;
}

extern "C" CDECL uint32_t
rand_next(rust_rng *rng) {
    return rng_gen_u32(rng);
}

extern "C" CDECL void
rand_free(rust_rng *rng) {
    rust_task *task = rust_get_current_task();
    task->free(rng);
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
    // Account for alignment. `front` may not indeed be the
    // front byte of the passed-in argument
    if (((uintptr_t)front % t->align) != 0) {
        front = (uint8_t *)align_to((uintptr_t)front, (size_t)t->align);
    }
    for (uintptr_t i = 0; i < t->size; ++front, ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, *front);
    }
}

extern "C" CDECL void
debug_box(type_desc *t, rust_opaque_box *box) {
    rust_task *task = rust_get_current_task();
    LOG(task, stdlib, "debug_box(0x%" PRIxPTR ")", box);
    debug_tydesc_helper(t);
    LOG(task, stdlib, "  refcount %" PRIdPTR,
        box->ref_count - 1);  // -1 because we ref'ed for this call
    uint8_t *data = (uint8_t *)box_body(box);
    for (uintptr_t i = 0; i < t->size; ++i) {
        LOG(task, stdlib, "  byte %" PRIdPTR ": 0x%" PRIx8, i, data[i]);
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

extern "C" CDECL void
debug_fn(type_desc *t, fn_env_pair *fn) {
    rust_task *task = rust_get_current_task();
    LOG(task, stdlib, "debug_fn");
    debug_tydesc_helper(t);
    LOG(task, stdlib, " fn at 0x%" PRIxPTR, fn->f);
    LOG(task, stdlib, "  env at 0x%" PRIxPTR, fn->env);
    if (fn->env) {
        LOG(task, stdlib, "    refcount %" PRIdPTR, fn->env->ref_count);
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

extern "C" CDECL char*
#if defined(__WIN32__)
rust_list_dir_val(WIN32_FIND_DATA* entry_ptr) {
    return entry_ptr->cFileName;
}
#else
rust_list_dir_val(dirent* entry_ptr) {
    return entry_ptr->d_name;
}
#endif

extern "C" CDECL size_t
#if defined(__WIN32__)
rust_list_dir_wfd_size() {
    return sizeof(WIN32_FIND_DATAW);
}
#else
rust_list_dir_wfd_size() {
    return 0;
}
#endif

extern "C" CDECL void*
#if defined(__WIN32__)
rust_list_dir_wfd_fp_buf(WIN32_FIND_DATAW* wfd) {
    if(wfd == NULL) {
        return 0;
    }
    else {
        return wfd->cFileName;
    }
}
#else
rust_list_dir_wfd_fp_buf(void* wfd) {
    return 0;
}
#endif

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

    const uint64_t NANOSECONDS_FROM_1601_TO_1970 = 11644473600000000ull;
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
        reserve_vec_exact(&out_tm->tm_zone, size + 1);
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
rust_gmtime(int64_t sec, int32_t nsec, rust_tm *timeptr) {
    tm tm;
    time_t s = sec;
    GMTIME(&s, &tm);

    tm_to_rust_tm(&tm, timeptr, 0, "UTC", nsec);
}

extern "C" CDECL void
rust_localtime(int64_t sec, int32_t nsec, rust_tm *timeptr) {
    tm tm;
    time_t s = sec;
    LOCALTIME(&s, &tm);

#if defined(__WIN32__)
    int32_t gmtoff = -timezone;
    char zone[64];
    strftime(zone, sizeof(zone), "%Z", &tm);
#else
    int32_t gmtoff = tm.tm_gmtoff;
    const char *zone = tm.tm_zone;
#endif

    tm_to_rust_tm(&tm, timeptr, gmtoff, zone, nsec);
}

extern "C" CDECL int64_t
rust_timegm(rust_tm* timeptr) {
    tm t;
    rust_tm_to_tm(timeptr, &t);
    return TIMEGM(&t);
}

extern "C" CDECL int64_t
rust_mktime(rust_tm* timeptr) {
    tm t;
    rust_tm_to_tm(timeptr, &t);
    return mktime(&t);
}

extern "C" CDECL rust_sched_id
rust_get_sched_id() {
    rust_task *task = rust_get_current_task();
    return task->sched->get_id();
}

extern "C" CDECL uintptr_t
rust_num_threads() {
    rust_task *task = rust_get_current_task();
    return task->kernel->env->num_sched_threads;
}

extern "C" CDECL int
rust_get_argc() {
    rust_task *task = rust_get_current_task();
    return task->kernel->env->argc;
}

extern "C" CDECL char**
rust_get_argv() {
    rust_task *task = rust_get_current_task();
    return task->kernel->env->argv;
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
    rust_sched_id sched_id = task->kernel->main_sched_id();
    rust_scheduler *sched = task->kernel->get_scheduler_by_id(sched_id);
    assert(sched != NULL && "should always have a main scheduler");
    return new_task_common(sched, task);
}

extern "C" CDECL rust_task*
rust_new_task_in_sched(rust_sched_id id) {
    rust_task *task = rust_get_current_task();
    rust_scheduler *sched = task->kernel->get_scheduler_by_id(id);
    if (sched == NULL)
        return NULL;
    return new_task_common(sched, task);
}

extern "C" rust_task *
rust_get_task() {
    return rust_get_current_task();
}

extern "C" rust_task *
rust_try_get_task() {
    return rust_try_get_current_task();
}

extern "C" CDECL stk_seg *
rust_get_stack_segment() {
    return rust_get_current_task()->stk;
}

extern "C" CDECL stk_seg *
rust_get_c_stack() {
    return rust_get_current_task()->get_c_stack();
}

extern "C" CDECL void
start_task(rust_task *target, fn_env_pair *f) {
    target->start(f->f, f->env, NULL);
}

extern "C" CDECL size_t
rust_sched_current_nonlazy_threads() {
    rust_task *task = rust_get_current_task();
    return task->sched->number_of_threads();
}

extern "C" CDECL size_t
rust_sched_threads() {
    rust_task *task = rust_get_current_task();
    return task->sched->max_number_of_threads();
}

// This is called by an intrinsic on the Rust stack and must run
// entirely in the red zone. Do not call on the C stack.
extern "C" CDECL MUST_CHECK bool
rust_task_yield(rust_task *task, bool *killed) {
    return task->yield();
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

extern "C" CDECL void
rust_dbg_breakpoint() {
    BREAKPOINT_AWESOME;
}

extern "C" CDECL rust_sched_id
rust_osmain_sched_id() {
    rust_task *task = rust_get_current_task();
    return task->kernel->osmain_sched_id();
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

extern "C" lock_and_signal*
rust_create_little_lock() {
    return new lock_and_signal();
}

extern "C" void
rust_destroy_little_lock(lock_and_signal *lock) {
    delete lock;
}

extern "C" void
rust_lock_little_lock(lock_and_signal *lock) {
    lock->lock();
}

extern "C" void
rust_unlock_little_lock(lock_and_signal *lock) {
    lock->unlock();
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

// set/get/atexit task_borrow_list can run on the rust stack for speed.
extern "C" void *
rust_take_task_borrow_list(rust_task *task) {
    void *r = task->borrow_list;
    task->borrow_list = NULL;
    return r;
}
extern "C" void
rust_set_task_borrow_list(rust_task *task, void *data) {
    assert(task->borrow_list == NULL);
    assert(data != NULL);
    task->borrow_list = data;
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

// Can safely run on the rust stack.
extern "C" void
rust_task_ref(rust_task *task) {
    task->ref();
}

// Don't run on the rust stack!
extern "C" void
rust_task_deref(rust_task *task) {
    task->deref();
}

// Must call on rust stack.
extern "C" CDECL void
rust_call_tydesc_glue(void *root, size_t *tydesc, size_t glue_index) {
    void (*glue_fn)(void *, void *, void *, void *) =
        (void (*)(void *, void *, void *, void *))tydesc[glue_index];
    if (glue_fn)
        glue_fn(0, 0, 0, root);
}

// Don't run on the Rust stack!
extern "C" void
rust_log_str(uint32_t level, const char *str, size_t size) {
    rust_task *task = rust_get_current_task();
    task->sched_loop->get_log().log(task, level, "%.*s", (int)size, str);
}

extern "C" CDECL void      record_sp_limit(void *limit);

class raw_thread: public rust_thread {
public:
    fn_env_pair fn;

    raw_thread(fn_env_pair fn) : fn(fn) { }

    virtual void run() {
        record_sp_limit(0);
        fn.f(NULL, fn.env, NULL);
    }
};

extern "C" raw_thread*
rust_raw_thread_start(fn_env_pair *fn) {
    assert(fn);
    raw_thread *thread = new raw_thread(*fn);
    thread->start();
    return thread;
}

extern "C" void
rust_raw_thread_join_delete(raw_thread *thread) {
    assert(thread);
    thread->join();
    delete thread;
}

extern "C" void
rust_register_exit_function(spawn_fn runner, fn_env_pair *f) {
    rust_task *task = rust_get_current_task();
    task->kernel->register_exit_function(runner, f);
}

extern "C" void *
rust_get_global_data_ptr() {
    rust_task *task = rust_get_current_task();
    return &task->kernel->global_data;
}

extern "C" void
rust_inc_kernel_live_count() {
    rust_task *task = rust_get_current_task();
    task->kernel->inc_live_count();
}

extern "C" void
rust_dec_kernel_live_count() {
    rust_task *task = rust_get_current_task();
    task->kernel->dec_live_count();
}

#ifndef _WIN32
#include <sys/types.h>
#include <dirent.h>

extern "C" DIR*
rust_opendir(char *dirname) {
    return opendir(dirname);
}

extern "C" dirent*
rust_readdir(DIR *dirp) {
    return readdir(dirp);
}

#else

extern "C" void
rust_opendir() {
}

extern "C" void
rust_readdir() {
}

#endif

extern "C" rust_env*
rust_get_rt_env() {
    rust_task *task = rust_get_current_task();
    return task->kernel->env;
}

#ifndef _WIN32
pthread_key_t rt_key = -1;
#else
DWORD rt_key = -1;
#endif

extern "C" void*
rust_get_rt_tls_key() {
    return &rt_key;
}

// Initialize the TLS key used by the new scheduler
extern "C" CDECL void
rust_initialize_rt_tls_key() {

    static lock_and_signal init_lock;
    static bool initialized = false;

    scoped_lock with(init_lock);

    if (!initialized) {

#ifndef _WIN32
        assert(!pthread_key_create(&rt_key, NULL));
#else
        rt_key = TlsAlloc();
        assert(rt_key != TLS_OUT_OF_INDEXES);
#endif

        initialized = true;
    }
}

extern "C" CDECL memory_region*
rust_new_memory_region(uintptr_t synchronized,
                       uintptr_t detailed_leaks,
                       uintptr_t poison_on_free) {
    return new memory_region((bool)synchronized,
                             (bool)detailed_leaks,
                             (bool)poison_on_free);
}

extern "C" CDECL void
rust_delete_memory_region(memory_region *region) {
    delete region;
}

extern "C" CDECL boxed_region*
rust_new_boxed_region(memory_region *region,
                      uintptr_t poison_on_free) {
    return new boxed_region(region, poison_on_free);
}

extern "C" CDECL void
rust_delete_boxed_region(boxed_region *region) {
    delete region;
}

extern "C" CDECL rust_opaque_box*
rust_boxed_region_malloc(boxed_region *region, type_desc *td, size_t size) {
    return region->malloc(td, size);
}

extern "C" CDECL void
rust_boxed_region_free(boxed_region *region, rust_opaque_box *box) {
    region->free(box);
}

typedef void *(rust_try_fn)(void*, void*);

extern "C" CDECL uintptr_t
rust_try(rust_try_fn f, void *fptr, void *env) {
    try {
        f(fptr, env);
    } catch (uintptr_t token) {
        assert(token != 0);
        return token;
    }
    return 0;
}

extern "C" CDECL void
rust_begin_unwind(uintptr_t token) {
#ifndef __WIN32__
    throw token;
#else
    abort();
#endif
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
