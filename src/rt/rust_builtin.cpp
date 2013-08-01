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

#include "rust_util.h"
#include "sync/timer.h"
#include "sync/rust_thread.h"
#include "sync/lock_and_signal.h"
#include "memory_region.h"
#include "boxed_region.h"
#include "rust_abi.h"
#include "rust_rng.h"
#include "vg/valgrind.h"
#include "sp.h"

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
    assert(seed != NULL);
    rust_rng *rng = (rust_rng *) malloc(sizeof(rust_rng));
    assert(rng != NULL && "rng alloc failed");
    rng_init(rng, NULL, seed, seed_size);
    return rng;
}

extern "C" CDECL uint32_t
rand_next(rust_rng *rng) {
    return rng_gen_u32(rng);
}

extern "C" CDECL void
rand_free(rust_rng *rng) {
    free(rng);
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

extern "C" int
debug_static_mut;

int debug_static_mut = 3;

extern "C" void
debug_static_mut_check_four() {
    assert(debug_static_mut == 4);
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
        memcpy(out_tm->tm_zone->data, zone, size);
        out_tm->tm_zone->fill = size + 1;
        out_tm->tm_zone->data[size] = '\0';
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

static lock_and_signal log_lock;
static bool log_to_console = true;

extern "C" CDECL void
rust_log_console_on() {
    scoped_lock with(log_lock);
    log_to_console = true;
}

extern "C" CDECL void
rust_log_console_off() {
    scoped_lock with(log_lock);
    log_to_console = false;
}

extern "C" CDECL uintptr_t
rust_should_log_console() {
    scoped_lock with(log_lock);
    return log_to_console;
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

class raw_thread: public rust_thread {
public:
    fn_env_pair fn;

    raw_thread(fn_env_pair fn) : fn(fn) { }

    virtual void run() {
        record_sp_limit(0);
        fn.f(fn.env, NULL);
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
rust_raw_thread_join(raw_thread *thread) {
    assert(thread);
    thread->join();
}

extern "C" void
rust_raw_thread_delete(raw_thread *thread) {
    assert(thread);
    delete thread;
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

extern "C" CDECL rust_opaque_box*
rust_boxed_region_realloc(boxed_region *region, rust_opaque_box *ptr, size_t size) {
    return region->realloc(ptr, size);
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

extern "C" CDECL uintptr_t
rust_running_on_valgrind() {
    return RUNNING_ON_VALGRIND;
}

extern int get_num_cpus();

extern "C" CDECL uintptr_t
rust_get_num_cpus() {
    return get_num_cpus();
}

static lock_and_signal global_args_lock;
static uintptr_t global_args_ptr = 0;

extern "C" CDECL void
rust_take_global_args_lock() {
    global_args_lock.lock();
}

extern "C" CDECL void
rust_drop_global_args_lock() {
    global_args_lock.unlock();
}

extern "C" CDECL uintptr_t*
rust_get_global_args_ptr() {
    return &global_args_ptr;
}

static lock_and_signal exit_status_lock;
static uintptr_t exit_status = 0;

extern "C" CDECL void
rust_set_exit_status_newrt(uintptr_t code) {
    scoped_lock with(exit_status_lock);
    exit_status = code;
}

extern "C" CDECL uintptr_t
rust_get_exit_status_newrt() {
    scoped_lock with(exit_status_lock);
    return exit_status;
}

static lock_and_signal change_dir_lock;

extern "C" CDECL void
rust_take_change_dir_lock() {
    change_dir_lock.lock();
}

extern "C" CDECL void
rust_drop_change_dir_lock() {
    change_dir_lock.unlock();
}

// Used by i386 __morestack
extern "C" CDECL uintptr_t
rust_get_task() {
    return 0;
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
