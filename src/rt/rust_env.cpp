// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The runtime wants to pull a number of variables out of the
// environment but calling getenv is not threadsafe, so every value
// that might come from the environment is loaded here, once, during
// init.

#include "sync/lock_and_signal.h"
#include "rust_env.h"

// The environment variables that the runtime knows about
#define RUST_THREADS "RUST_THREADS"
#define RUST_MIN_STACK "RUST_MIN_STACK"
#define RUST_MAX_STACK "RUST_MAX_STACK"
#define RUST_LOG "RUST_LOG"
#define DETAILED_LEAKS "DETAILED_LEAKS"
#define RUST_SEED "RUST_SEED"
#define RUST_POISON_ON_FREE "RUST_POISON_ON_FREE"
#define RUST_DEBUG_MEM "RUST_DEBUG_MEM"
#define RUST_DEBUG_BORROW "RUST_DEBUG_BORROW"

static lock_and_signal env_lock;

extern "C" CDECL void
rust_take_env_lock() {
    env_lock.lock();
}

extern "C" CDECL void
rust_drop_env_lock() {
    env_lock.unlock();
}

#if defined(__WIN32__)
static int
get_num_cpus() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);

    return (int) sysinfo.dwNumberOfProcessors;
}
#elif defined(__BSD__)
static int
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
static int
get_num_cpus() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}
#endif

static int
get_num_threads()
{
    char *env = getenv(RUST_THREADS);
    if(env) {
        int num = atoi(env);
        if(num > 0)
            return num;
    }
    return get_num_cpus();
}

static size_t
get_min_stk_size() {
    char *minsz = getenv(RUST_MIN_STACK);
    if(minsz) {
        return strtol(minsz, NULL, 0);
    }
    else {
        return 0x300;
    }
}

static size_t
get_max_stk_size() {
    char *maxsz = getenv(RUST_MAX_STACK);
    if (maxsz) {
        return strtol(maxsz, NULL, 0);
    }
    else {
        return 1024*1024*1024;
    }
}

static char*
copyenv(const char* name) {
    char *envvar = getenv(name);
    if (!envvar) {
        return NULL;
    } else {
        size_t slen = strlen(envvar);
        size_t buflen = slen + 1;
        char *var = (char*)malloc(buflen);
        memset(var, 0, buflen);
        strncpy(var, envvar, slen);
        return var;
    }
}

rust_env*
load_env(int argc, char **argv) {
    scoped_lock with(env_lock);

    rust_env *env = (rust_env*)malloc(sizeof(rust_env));

    env->num_sched_threads = (size_t)get_num_threads();
    env->min_stack_size = get_min_stk_size();
    env->max_stack_size = get_max_stk_size();
    env->logspec = copyenv(RUST_LOG);
    env->detailed_leaks = getenv(DETAILED_LEAKS) != NULL;
    env->rust_seed = copyenv(RUST_SEED);
    env->poison_on_free = getenv(RUST_POISON_ON_FREE) != NULL;
    env->argc = argc;
    env->argv = argv;
    env->debug_mem = getenv(RUST_DEBUG_MEM) != NULL;
    env->debug_borrow = getenv(RUST_DEBUG_BORROW) != NULL;
    return env;
}

void
free_env(rust_env *env) {
    free(env->logspec);
    free(env->rust_seed);
    free(env);
}

