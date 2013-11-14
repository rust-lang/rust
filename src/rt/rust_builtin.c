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

#include "rust_globals.h"
#include "vg/valgrind.h"

#include <time.h>

#ifdef __APPLE__
    #include <TargetConditionals.h>
    #include <mach/mach_time.h>

    #if (TARGET_OS_IPHONE)
        extern char **environ;
    #else
        #include <crt_externs.h>
    #endif
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
    if (tz)
        tz = strdup(tz);
    setenv("TZ", "", 1);
    tzset();
    ret = mktime(tm);
    if (tz) {
        setenv("TZ", tz, 1);
        free(tz);
    } else
        unsetenv("TZ");
    tzset();
    return ret;
}
#endif

#if defined(__WIN32__)
char**
rust_env_pairs() {
    return 0;
}
#else
char**
rust_env_pairs() {
#if defined(__APPLE__) && !(TARGET_OS_IPHONE)
    char **environ = *_NSGetEnviron();
#endif
    return environ;
}
#endif

char*
#if defined(__WIN32__)
rust_list_dir_val(WIN32_FIND_DATA* entry_ptr) {
    return entry_ptr->cFileName;
}
#else
rust_list_dir_val(struct dirent* entry_ptr) {
    return entry_ptr->d_name;
}
#endif

size_t
#if defined(__WIN32__)
rust_list_dir_wfd_size() {
    return sizeof(WIN32_FIND_DATAW);
}
#else
rust_list_dir_wfd_size() {
    return 0;
}
#endif

void*
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

int
rust_path_is_dir(const char *path) {
    struct stat buf;
    if (stat(path, &buf)) {
        return 0;
    }
    return S_ISDIR(buf.st_mode);
}

int
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

int
rust_path_exists(const char *path) {
    struct stat buf;
    if (stat(path, &buf)) {
        return 0;
    }
    return 1;
}

int
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

FILE* rust_get_stdin() {return stdin;}
FILE* rust_get_stdout() {return stdout;}
FILE* rust_get_stderr() {return stderr;}

#if defined(__WIN32__)
void
rust_get_time(int64_t *sec, int32_t *nsec) {
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
void
rust_get_time(int64_t *sec, int32_t *nsec) {
#ifdef __APPLE__
    struct timeval tv;
    gettimeofday(&tv, NULL);
    *sec = tv.tv_sec;
    *nsec = tv.tv_usec * 1000;
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    *sec = ts.tv_sec;
    *nsec = ts.tv_nsec;
#endif
}
#endif

const int64_t ns_per_s = 1000000000LL;

void
rust_precise_time_ns(uint64_t *ns) {

#ifdef __APPLE__
    uint64_t time = mach_absolute_time();
    mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t time_nano = time * (info.numer / info.denom);
    *ns = time_nano;
#elif __WIN32__
    LARGE_INTEGER ticks_per_s;
    BOOL query_result = QueryPerformanceFrequency(&ticks_per_s);
    assert(query_result);
    if (ticks_per_s.QuadPart == 0LL) {
        ticks_per_s.QuadPart = 1LL;
    }
    LARGE_INTEGER ticks;
    query_result = QueryPerformanceCounter(&ticks);
    assert(query_result);
    *ns = (uint64_t)((ticks.QuadPart * ns_per_s) / ticks_per_s.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    *ns = (uint64_t)(ts.tv_sec * ns_per_s + ts.tv_nsec);
#endif
}

typedef struct
{
    size_t fill;    // in bytes; if zero, heapified
    size_t alloc;   // in bytes
    uint8_t data[0];
} rust_vec;

typedef rust_vec rust_str;

typedef struct {
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
} rust_tm;

void rust_tm_to_tm(rust_tm* in_tm, struct tm* out_tm) {
    memset(out_tm, 0, sizeof(struct tm));
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

void tm_to_rust_tm(struct tm* in_tm, rust_tm* out_tm, int32_t gmtoff,
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
struct tm* GMTIME(const time_t *clock, struct tm *result) {
    struct tm* t = gmtime(clock);
    if (t == NULL || result == NULL) { return NULL; }
    *result = *t;
    return result;
}
struct tm* LOCALTIME(const time_t *clock, struct tm *result) {
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

void
rust_tzset() {
    TZSET();
}

void
rust_gmtime(int64_t sec, int32_t nsec, rust_tm *timeptr) {
    struct tm tm;
    time_t s = sec;
    GMTIME(&s, &tm);

    tm_to_rust_tm(&tm, timeptr, 0, "UTC", nsec);
}

void
rust_localtime(int64_t sec, int32_t nsec, rust_tm *timeptr) {
    struct tm tm;
    time_t s = sec;
    LOCALTIME(&s, &tm);

    const char* zone = NULL;
#if defined(__WIN32__)
    int32_t gmtoff = -timezone;
    wchar_t wbuffer[64] = {0};
    char buffer[256] = {0};
    // strftime("%Z") can contain non-UTF-8 characters on non-English locale (issue #9418),
    // so time zone should be converted from UTF-16 string.
    // Since wcsftime depends on setlocale() result,
    // instead we convert it using MultiByteToWideChar.
    if (strftime(buffer, sizeof(buffer) / sizeof(char), "%Z", &tm) > 0) {
        // ANSI -> UTF-16
        MultiByteToWideChar(CP_ACP, 0, buffer, -1, wbuffer, sizeof(wbuffer) / sizeof(wchar_t));
        // UTF-16 -> UTF-8
        WideCharToMultiByte(CP_UTF8, 0, wbuffer, -1, buffer, sizeof(buffer), NULL, NULL);
        zone = buffer;
    }
#else
    int32_t gmtoff = tm.tm_gmtoff;
    zone = tm.tm_zone;
#endif

    tm_to_rust_tm(&tm, timeptr, gmtoff, zone, nsec);
}

int64_t
rust_timegm(rust_tm* timeptr) {
    struct tm t;
    rust_tm_to_tm(timeptr, &t);
    return TIMEGM(&t);
}

int64_t
rust_mktime(rust_tm* timeptr) {
    struct tm t;
    rust_tm_to_tm(timeptr, &t);
    return mktime(&t);
}

#ifndef _WIN32
#include <sys/types.h>
#include <dirent.h>

DIR*
rust_opendir(char *dirname) {
    return opendir(dirname);
}

struct dirent*
rust_readdir(DIR *dirp) {
    return readdir(dirp);
}

#else

void
rust_opendir() {
}

void
rust_readdir() {
}

#endif

uintptr_t
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

uintptr_t
rust_get_num_cpus() {
    return get_num_cpus();
}

unsigned int
rust_valgrind_stack_register(void *start, void *end) {
  return VALGRIND_STACK_REGISTER(start, end);
}

void
rust_valgrind_stack_deregister(unsigned int id) {
  VALGRIND_STACK_DEREGISTER(id);
}

#if defined(__WIN32__)

void
rust_unset_sigprocmask() {
    // empty stub for windows to keep linker happy
}

#else

#include <signal.h>
#include <unistd.h>

void
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

void
rust_win32_rand_acquire(HCRYPTPROV* phProv) {
    win32_require
        (_T("CryptAcquireContext"),
         // changes to the parameters here should be reflected in the docs of
         // std::rand::os::OSRng
         CryptAcquireContext(phProv, NULL, NULL, PROV_RSA_FULL,
                             CRYPT_VERIFYCONTEXT|CRYPT_SILENT));

}
void
rust_win32_rand_gen(HCRYPTPROV hProv, DWORD dwLen, BYTE* pbBuffer) {
    win32_require
        (_T("CryptGenRandom"), CryptGenRandom(hProv, dwLen, pbBuffer));
}
void
rust_win32_rand_release(HCRYPTPROV hProv) {
    win32_require
        (_T("CryptReleaseContext"), CryptReleaseContext(hProv, 0));
}

#else

// these symbols are listed in rustrt.def.in, so they need to exist; but they
// should never be called.

void
rust_win32_rand_acquire() {
    abort();
}
void
rust_win32_rand_gen() {
    abort();
}
void
rust_win32_rand_release() {
    abort();
}

#endif

#if defined(__WIN32__)

int
rust_crit_section_size() { return sizeof(CRITICAL_SECTION); }
int
rust_pthread_mutex_t_size() { return 0; }
int
rust_pthread_cond_t_size() { return 0; }

#else

int
rust_crit_section_size() { return 0; }
int
rust_pthread_mutex_t_size() { return sizeof(pthread_mutex_t); }
int
rust_pthread_cond_t_size() { return sizeof(pthread_cond_t); }

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
