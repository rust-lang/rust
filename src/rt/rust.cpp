#include "rust_internal.h"

struct
command_line_args : public kernel_owned<command_line_args>
{
    rust_kernel *kernel;
    rust_task *task;
    int argc;
    char **argv;

    // vec[str] passed to rust_task::start.
    rust_vec *args;

    command_line_args(rust_task *task,
                      int sys_argc,
                      char **sys_argv)
        : kernel(task->kernel),
          task(task),
          argc(sys_argc),
          argv(sys_argv),
          args(NULL)
    {
#if defined(__WIN32__)
        LPCWSTR cmdline = GetCommandLineW();
        LPWSTR *wargv = CommandLineToArgvW(cmdline, &argc);
        kernel->win32_require("CommandLineToArgvW", wargv != NULL);
        argv = (char **) kernel->malloc(sizeof(char*) * argc);
        for (int i = 0; i < argc; ++i) {
            int n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                              NULL, 0, NULL, NULL);
            kernel->win32_require("WideCharToMultiByte(0)", n_chars != 0);
            argv[i] = (char *) kernel->malloc(n_chars);
            n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                          argv[i], n_chars, NULL, NULL);
            kernel->win32_require("WideCharToMultiByte(1)", n_chars != 0);
        }
        LocalFree(wargv);
#endif
        size_t vec_fill = sizeof(rust_str *) * argc;
        size_t vec_alloc = next_power_of_two(sizeof(rust_vec) + vec_fill);
        void *mem = kernel->malloc(vec_alloc);
        args = new (mem) rust_vec(task->sched, vec_alloc, 0, NULL);
        rust_str **strs = (rust_str**) &args->data[0];
        for (int i = 0; i < argc; ++i) {
            size_t str_fill = strlen(argv[i]) + 1;
            size_t str_alloc = next_power_of_two(sizeof(rust_str) + str_fill);
            mem = kernel->malloc(str_alloc);
            strs[i] = new (mem) rust_str(task->sched, str_alloc, str_fill,
                                         (uint8_t const *)argv[i]);
        }
        args->fill = vec_fill;
        // If the caller has a declared args array, they may drop; but
        // we don't know if they have such an array. So we pin the args
        // array here to ensure it survives to program-shutdown.
        args->ref();
    }

    ~command_line_args() {
        if (args) {
            // Drop the args we've had pinned here.
            rust_str **strs = (rust_str**) &args->data[0];
            for (int i = 0; i < argc; ++i)
                kernel->free(strs[i]);
            kernel->free(args);
        }

#ifdef __WIN32__
        for (int i = 0; i < argc; ++i) {
            kernel->free(argv[i]);
        }
        kernel->free(argv);
#endif
    }
};

int get_num_threads()
{
    char *env = getenv("RUST_THREADS");
    if(env) {
        int num = atoi(env);
        if(num > 0)
            return num;
    }
    // FIXME: in this case, determine the number of CPUs present on the
    // machine.
    return 1;
}

/**
 * Main entry point into the Rust runtime. Here we create a Rust service,
 * initialize the kernel, create the root domain and run it.
 */

int check_claims = 0;

void enable_claims(void* ck) { check_claims = (ck != 0); }

extern "C" CDECL int
rust_start(uintptr_t main_fn, int argc, char **argv, void* crate_map) {

    update_log_settings(crate_map, getenv("RUST_LOG"));
    enable_claims(getenv("CHECK_CLAIMS"));

    rust_srv *srv = new rust_srv();
    rust_kernel *kernel = new rust_kernel(srv);
    kernel->start();
    rust_scheduler *sched = kernel->get_scheduler();
    command_line_args *args
        = new (kernel) command_line_args(sched->root_task, argc, argv);

    DLOG(sched, dom, "startup: %d args in 0x%" PRIxPTR,
             args->argc, (uintptr_t)args->args);
    for (int i = 0; i < args->argc; i++) {
        DLOG(sched, dom, "startup: arg[%d] = '%s'", i, args->argv[i]);
    }

    sched->root_task->start(main_fn, (uintptr_t)args->args);

    int num_threads = get_num_threads();

    DLOG(sched, dom, "Using %d worker threads.", num_threads);

    int ret = kernel->start_task_threads(num_threads);
    delete args;
    delete kernel;
    delete srv;

#if !defined(__WIN32__)
    // Don't take down the process if the main thread exits without an
    // error.
    if (!ret)
        pthread_exit(NULL);
#endif
    return ret;
}

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
