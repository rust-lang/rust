#include "rust_internal.h"

struct
command_line_args : public kernel_owned<command_line_args>
{
    rust_kernel *kernel;
    rust_task *task;
    int argc;
    char **argv;

    // [str] passed to rust_task::start.
    rust_vec *args;

    command_line_args(rust_task *task,
                      int sys_argc,
                      char **sys_argv)
        : kernel(task->kernel),
          task(task),
          argc(sys_argc),
          argv(sys_argv)
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

        args = (rust_vec *)
            kernel->malloc(vec_size<rust_vec*>(argc),
                           "command line arg interior");
        args->fill = args->alloc = sizeof(rust_vec*) * argc;
        for (int i = 0; i < argc; ++i) {
            rust_str *str = make_str(kernel, argv[i],
                                     strlen(argv[i]),
                                     "command line arg");
            ((rust_str**)&args->data)[i] = str;
        }
    }

    ~command_line_args() {
        for (int i = 0; i < argc; ++i) {
            rust_vec *s = ((rust_vec**)&args->data)[i];
            kernel->free(s);
        }
        kernel->free(args);

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
rust_start(uintptr_t main_fn, int argc, char **argv,
           void* crate_map) {

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
    root_task->deref();
    root_task = NULL;

    int ret = kernel->start_task_threads();
    delete args;
    delete kernel;
    delete srv;

    free_env(env);

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
