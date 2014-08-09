// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#include <stdint.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#if !defined(__WIN32__)
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#else
#include <windows.h>
#include <wincrypt.h>
#include <stdio.h>
#include <tchar.h>
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <mach/mach_time.h>

#if !(TARGET_OS_IPHONE)
#include <crt_externs.h>
#endif
#endif

/* Foreign builtins. */
//include valgrind.h after stdint.h so that uintptr_t is defined for msys2 w64
#include "valgrind/valgrind.h"

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

#ifdef __APPLE__
#if (TARGET_OS_IPHONE)
extern char **environ;
#endif
#endif

#if defined(__FreeBSD__) || defined(__linux__) || defined(__ANDROID__) || defined(__DragonFly__)
extern char **environ;
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

void tm_to_rust_tm(struct tm* in_tm,
                   rust_tm* out_tm,
                   int32_t gmtoff,
                   int32_t nsec) {
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

    tm_to_rust_tm(&tm, timeptr, 0, nsec);
}

void
rust_localtime(int64_t sec, int32_t nsec, rust_tm *timeptr) {
    struct tm tm;
    time_t s = sec;
    LOCALTIME(&s, &tm);

#if defined(__WIN32__)
    int32_t gmtoff = -timezone;
#else
    int32_t gmtoff = tm.tm_gmtoff;
#endif

    tm_to_rust_tm(&tm, timeptr, gmtoff, nsec);
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

DIR*
rust_opendir(char *dirname) {
    return opendir(dirname);
}

int
rust_readdir_r(DIR *dirp, struct dirent *entry, struct dirent **result) {
    return readdir_r(dirp, entry, result);
}

int
rust_dirent_t_size() {
    return sizeof(struct dirent);
}

#else

void
rust_opendir() {
}

void
rust_readdir() {
}

void
rust_dirent_t_size() {
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

void
rust_unset_sigprocmask() {
    // this can't be safely converted to rust code because the
    // representation of sigset_t is platform-dependent
    sigset_t sset;
    sigemptyset(&sset);
    sigprocmask(SIG_SETMASK, &sset, NULL);
}

#endif

#if defined(__DragonFly__)
#include <errno.h>
// In DragonFly __error() is an inline function and as such
// no symbol exists for it.
int *__dfly_error(void) { return __error(); }
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
