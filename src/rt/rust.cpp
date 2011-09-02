#include "rust_internal.h"

struct
command_line_args : public kernel_owned<command_line_args>
{
    rust_kernel *kernel;
    rust_task *task;
    int argc;
    char **argv;
    rust_str **strs;

    // [str] passed to rust_task::start.
    rust_vec *args;
    rust_vec *args_istr;

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

        // Allocate a vector of estrs
        size_t vec_fill = sizeof(rust_str *) * argc;
        size_t vec_alloc = next_power_of_two(vec_fill);
        void *mem = kernel->malloc(vec_alloc, "command line");
        strs = (rust_str**) mem;
        for (int i = 0; i < argc; ++i) {
            size_t str_fill = strlen(argv[i]) + 1;
            size_t str_alloc = next_power_of_two(sizeof(rust_str) + str_fill);
            mem = kernel->malloc(str_alloc, "command line arg");
            strs[i] = new (mem) rust_str(str_alloc, str_fill,
                                         (uint8_t const *)argv[i]);
            strs[i]->ref_count++;
        }

        args = (rust_vec *)
            kernel->malloc(vec_size<rust_str*>(argc),
                           "command line arg interior");
        args->fill = args->alloc = sizeof(rust_str *) * argc;
        memcpy(&args->data[0], strs, args->fill);

        // Allocate a vector of istrs
        args_istr = (rust_vec *)
            kernel->malloc(vec_size<rust_vec*>(argc),
                           "command line arg interior");
        args_istr->fill = args_istr->alloc = sizeof(rust_vec*) * argc;
        for (int i = 0; i < argc; ++i) {
            rust_vec *str = make_istr(kernel, argv[i],
                                      strlen(argv[i]),
                                      "command line arg");
            ((rust_vec**)&args_istr->data)[i] = str;
        }
    }

    ~command_line_args() {
        // Free the estr args
        kernel->free(args);
        for (int i = 0; i < argc; ++i)
            kernel->free(strs[i]);
        kernel->free(strs);

        // Free the istr args
        for (int i = 0; i < argc; ++i) {
            rust_vec *s = ((rust_vec**)&args_istr->data)[i];
            kernel->free(s);
        }
        kernel->free(args_istr);

#ifdef __WIN32__
        for (int i = 0; i < argc; ++i) {
            kernel->free(argv[i]);
        }
        kernel->free(argv);
#endif
    }
};


// FIXME: Transitional. Please remove.
bool main_takes_istr = false;

extern "C" CDECL void
set_main_takes_istr(uintptr_t flag) {
    main_takes_istr = flag != 0;
}

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

    if (main_takes_istr) {
        root_task->start(main_fn, (uintptr_t)args->args_istr);
    } else {
        root_task->start(main_fn, (uintptr_t)args->args);
    }
    root_task->deref();
    root_task = NULL;

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
