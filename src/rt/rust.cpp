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
        argv = (char **) kernel->malloc(sizeof(char*) * argc,
                                        "win32 command line");
        for (int i = 0; i < argc; ++i) {
            int n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                              NULL, 0, NULL, NULL);
            kernel->win32_require("WideCharToMultiByte(0)", n_chars != 0);
            argv[i] = (char *) kernel->malloc(n_chars,
                                              "win32 command line arg");
            n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                          argv[i], n_chars, NULL, NULL);
            kernel->win32_require("WideCharToMultiByte(1)", n_chars != 0);
        }
        LocalFree(wargv);
#endif
        size_t vec_fill = sizeof(rust_str *) * argc;
        size_t vec_alloc = next_power_of_two(sizeof(rust_vec) + vec_fill);
        void *mem = kernel->malloc(vec_alloc, "command line");
        args = new (mem) rust_vec(vec_alloc, 0, NULL);
        rust_str **strs = (rust_str**) &args->data[0];
        for (int i = 0; i < argc; ++i) {
            size_t str_fill = strlen(argv[i]) + 1;
            size_t str_alloc = next_power_of_two(sizeof(rust_str) + str_fill);
            mem = kernel->malloc(str_alloc, "command line arg");
            strs[i] = new (mem) rust_str(str_alloc, str_fill,
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


/**
 * Main entry point into the Rust runtime. Here we create a Rust service,
 * initialize the kernel, create the root domain and run it.
 */

int check_claims = 0;

extern "C" CDECL int
rust_start(uintptr_t main_fn, int argc, char **argv, void* crate_map) {

    rust_env *env = load_env();

    update_log_settings(crate_map, env->logspec);
    check_claims = env->check_claims;

    rust_srv *srv = new rust_srv(env);
    rust_kernel *kernel = new rust_kernel(srv, env->num_sched_threads);
    rust_task_id root_id = kernel->create_task(NULL, "main");
    rust_task *root_task = kernel->get_task_by_id(root_id);
    I(kernel, root_task != NULL);
    rust_scheduler *sched = root_task->sched;
    command_line_args *args
        = new (kernel, "main command line args")
        command_line_args(root_task, argc, argv);

    DLOG(sched, dom, "startup: %d args in 0x%" PRIxPTR,
             args->argc, (uintptr_t)args->args);
    for (int i = 0; i < args->argc; i++) {
        DLOG(sched, dom, "startup: arg[%d] = '%s'", i, args->argv[i]);
    }

    root_task->start(main_fn, (uintptr_t)args->args);

    int ret = kernel->start_task_threads();
    delete args;
    delete kernel;
    delete srv;

    free_env(env);

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
