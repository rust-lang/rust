// The runtime wants to pull a number of variables out of the
// environment but calling getenv is not threadsafe, so every value
// that might come from the environment is loaded here, once, during
// init.

#include "rust_internal.h"

// The environment variables that the runtime knows about
#define RUST_THREADS "RUST_THREADS"
#define RUST_MIN_STACK "RUST_MIN_STACK"
#define RUST_LOG "RUST_LOG"
#define CHECK_CLAIMS "CHECK_CLAIMS"
#define DETAILED_LEAKS "DETAILED_LEAKS"
#define RUST_SEED "RUST_SEED"

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

// FIXME (issue #151): This should be 0x300; the change here is for
// practicality's sake until stack growth is working.

static size_t
get_min_stk_size() {
    char *stack_size = getenv(RUST_MIN_STACK);
    if(stack_size) {
        return strtol(stack_size, NULL, 0);
    }
    else {
        return 0x300000;
    }
}

static char*
copyenv(const char* name) {
    char *envvar = getenv(name);
    if (!envvar) {
        return NULL;
    } else {
        const size_t maxlen = 4096;
        size_t strlen = strnlen(envvar, maxlen);
        size_t buflen = strlen + 1;
        char *var = (char*)malloc(buflen);
        memset(var, 0, buflen);
        strncpy(var, envvar, strlen);
        return var;
    }
}

rust_env*
load_env() {
    rust_env *env = (rust_env*)malloc(sizeof(rust_env));

    env->num_sched_threads = (size_t)get_num_threads();
    env->min_stack_size = get_min_stk_size();
    env->logspec = copyenv(RUST_LOG);
    env->check_claims = getenv(CHECK_CLAIMS) != NULL;
    env->detailed_leaks = getenv(DETAILED_LEAKS) != NULL;
    env->rust_seed = copyenv(RUST_SEED);

    return env;
}

void
free_env(rust_env *env) {
    free(env->logspec);
    free(env->rust_seed);
    free(env);
}
