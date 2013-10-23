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
#include "sync/rust_thread.h"
#include "sync/lock_and_signal.h"
#include "memory_region.h"
#include "boxed_region.h"
#include "vg/valgrind.h"

#include <time.h>

#ifdef __APPLE__
#include <crt_externs.h>
#include <mach/mach_time.h>
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
rust_path_is_dir(const char *path) {
    struct stat buf;
    if (stat(path, &buf)) {
        return 0;
    }
    return S_ISDIR(buf.st_mode);
}

extern "C" CDECL int
#if defined(__WIN32__)
rust_path_is_dir_u16(const wchar_t *path) {
    struct _stat buf;
    // Don't use GetFileAttributesW, it cannot get attributes of
    // some system files (e.g. pagefile.sys).
    if (_wstat(path, &buf)) {
        return 0;
    }
    return S_ISDIR(buf.st_mode);
}
#else
rust_path_is_dir_u16(const void *path) {
    // Wide version of function is only used on Windows.
    return 0;
}
#endif

extern "C" CDECL int
rust_path_exists(const char *path) {
    struct stat buf;
    if (stat(path, &buf)) {
        return 0;
    }
    return 1;
}

extern "C" CDECL int
#if defined(__WIN32__)
rust_path_exists_u16(const wchar_t *path) {
    struct _stat buf;
    if (_wstat(path, &buf)) {
        return 0;
    }
    return 1;
}
#else
rust_path_exists_u16(const void *path) {
    // Wide version of function is only used on Windows.
    return 0;
}
#endif

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

const uint64_t ns_per_s = 1000000000LL;

extern "C" CDECL void
precise_time_ns(uint64_t *ns) {

#ifdef __APPLE__
    uint64_t time = mach_absolute_time();
    mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t time_nano = time * (info.numer / info.denom);
    *ns = time_nano;
#elif __WIN32__
    uint64_t ticks_per_s;
    QueryPerformanceFrequency((LARGE_INTEGER *)&ticks_per_s);
    if (ticks_per_s == 0LL) {
        ticks_per_s = 1LL;
    }
    uint64_t ticks;
    QueryPerformanceCounter((LARGE_INTEGER *)&ticks);
    *ns = ((ticks * ns_per_s) / ticks_per_s);
#else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *ns = (ts.tv_sec * ns_per_s + ts.tv_nsec);
#endif
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
        assert(out_tm->tm_zone->alloc >= size);
        memcpy(out_tm->tm_zone->data, zone, size);
        out_tm->tm_zone->fill = size;
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

    const char* zone = NULL;
#if defined(__WIN32__)
    int32_t gmtoff = -timezone;
    wchar_t wbuffer[64];
    char buffer[256];
    // strftime("%Z") can contain non-UTF-8 characters on non-English locale (issue #9418),
    // so time zone should be converted from UTF-16 string set by wcsftime.
    if (wcsftime(wbuffer, sizeof(wbuffer) / sizeof(wchar_t), L"%Z", &tm) > 0) {
        WideCharToMultiByte(CP_UTF8, 0, wbuffer, -1, buffer, sizeof(buffer), NULL, NULL);
        zone = buffer;
    }
#else
    int32_t gmtoff = tm.tm_gmtoff;
    zone = tm.tm_zone;
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

typedef void(startfn)(void*, void*);

class raw_thread: public rust_thread {
public:
    startfn *raw_start;
    void *rust_fn;
    void *rust_env;

    raw_thread(startfn *raw_start, void *rust_fn, void *rust_env)
        : raw_start(raw_start), rust_fn(rust_fn), rust_env(rust_env) { }

    virtual void run() {
        raw_start(rust_fn, rust_env);
    }
};

extern "C" raw_thread*
rust_raw_thread_start(startfn *raw_start, void *rust_start, void *rust_env) {
    assert(raw_start && rust_start);
    raw_thread *thread = new raw_thread(raw_start, rust_start, rust_env);
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
typedef pthread_key_t tls_key;
#else
typedef DWORD tls_key;
#endif

// Initialize the TLS key used by the new scheduler
extern "C" CDECL void
rust_initialize_rt_tls_key(tls_key *key) {

    static lock_and_signal init_lock;
    static bool initialized = false;

    scoped_lock with(init_lock);

    if (!initialized) {

#ifndef _WIN32
        assert(!pthread_key_create(key, NULL));
#else
        *key = TlsAlloc();
        assert(*key != TLS_OUT_OF_INDEXES);
#endif

        initialized = true;
    }
}

extern "C" CDECL memory_region*
rust_new_memory_region(uintptr_t detailed_leaks,
                       uintptr_t poison_on_free) {
    return new memory_region((bool)detailed_leaks,
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
    throw token;
}

extern "C" CDECL uintptr_t
rust_running_on_valgrind() {
    return RUNNING_ON_VALGRIND;
}

#if defined(__WIN32__)
int
get_num_cpus() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);

    return (int) sysinfo.dwNumberOfProcessors;
}
#elif defined(__BSD__)
int
get_num_cpus() {
    /* swiped from http://stackoverflow.com/questions/150355/
       programmatically-find-the-number-of-cores-on-a-machine */

    unsigned int numCPU;
    int mib[4];
    size_t len = sizeof(numCPU);

    /* set the mib for hw.ncpu */
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

    /* get the number of CPUs from the system */
    sysctl(mib, 2, &numCPU, &len, NULL, 0);

    if( numCPU < 1 ) {
        mib[1] = HW_NCPU;
        sysctl( mib, 2, &numCPU, &len, NULL, 0 );

        if( numCPU < 1 ) {
            numCPU = 1;
        }
    }
    return numCPU;
}
#elif defined(__GNUC__)
int
get_num_cpus() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}
#endif

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

static lock_and_signal env_lock;

extern "C" CDECL void
rust_take_env_lock() {
    env_lock.lock();
}

extern "C" CDECL void
rust_drop_env_lock() {
    env_lock.unlock();
}

static lock_and_signal dlerror_lock;

extern "C" CDECL void
rust_take_dlerror_lock() {
    dlerror_lock.lock();
}

extern "C" CDECL void
rust_drop_dlerror_lock() {
    dlerror_lock.unlock();
}

extern "C" CDECL unsigned int
rust_valgrind_stack_register(void *start, void *end) {
  return VALGRIND_STACK_REGISTER(start, end);
}

extern "C" CDECL void
rust_valgrind_stack_deregister(unsigned int id) {
  VALGRIND_STACK_DEREGISTER(id);
}

#if defined(__WIN32__)

extern "C" CDECL void
rust_unset_sigprocmask() {
    // empty stub for windows to keep linker happy
}

#else

#include <signal.h>
#include <unistd.h>

extern "C" CDECL void
rust_unset_sigprocmask() {
    // this can't be safely converted to rust code because the
    // representation of sigset_t is platform-dependent
    sigset_t sset;
    sigemptyset(&sset);
    sigprocmask(SIG_SETMASK, &sset, NULL);
}

#endif

#if defined(__WIN32__)
void
win32_require(LPCTSTR fn, BOOL ok) {
    if (!ok) {
        LPTSTR buf;
        DWORD err = GetLastError();
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                      FORMAT_MESSAGE_FROM_SYSTEM |
                      FORMAT_MESSAGE_IGNORE_INSERTS,
                      NULL, err,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &buf, 0, NULL );
        fprintf(stderr, "%s failed with error %ld: %s", fn, err, buf);
        LocalFree((HLOCAL)buf);
        abort();
    }
}

extern "C" CDECL void
rust_win32_rand_acquire(HCRYPTPROV* phProv) {
    win32_require
        (_T("CryptAcquireContext"),
         // changes to the parameters here should be reflected in the docs of
         // std::rand::os::OSRng
         CryptAcquireContext(phProv, NULL, NULL, PROV_RSA_FULL,
                             CRYPT_VERIFYCONTEXT|CRYPT_SILENT));

}
extern "C" CDECL void
rust_win32_rand_gen(HCRYPTPROV hProv, DWORD dwLen, BYTE* pbBuffer) {
    win32_require
        (_T("CryptGenRandom"), CryptGenRandom(hProv, dwLen, pbBuffer));
}
extern "C" CDECL void
rust_win32_rand_release(HCRYPTPROV hProv) {
    win32_require
        (_T("CryptReleaseContext"), CryptReleaseContext(hProv, 0));
}

#else

// these symbols are listed in rustrt.def.in, so they need to exist; but they
// should never be called.

extern "C" CDECL void
rust_win32_rand_acquire() {
    abort();
}
extern "C" CDECL void
rust_win32_rand_gen() {
    abort();
}
extern "C" CDECL void
rust_win32_rand_release() {
    abort();
}

#endif
//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
