#include "rust_internal.h"

struct
command_line_args : public dom_owned<command_line_args>
{
    rust_dom *dom;
    int argc;
    char **argv;

    // vec[str] passed to rust_task::start.
    rust_vec *args;

    command_line_args(rust_dom *dom,
                      int sys_argc,
                      char **sys_argv)
        : dom(dom),
          argc(sys_argc),
          argv(sys_argv),
          args(NULL)
    {
#if defined(__WIN32__)
        LPCWSTR cmdline = GetCommandLineW();
        LPWSTR *wargv = CommandLineToArgvW(cmdline, &argc);
        dom->win32_require("CommandLineToArgvW", wargv != NULL);
        argv = (char **) dom->malloc(sizeof(char*) * argc);
        for (int i = 0; i < argc; ++i) {
            int n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                              NULL, 0, NULL, NULL);
            dom->win32_require("WideCharToMultiByte(0)", n_chars != 0);
            argv[i] = (char *) dom->malloc(n_chars);
            n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                          argv[i], n_chars, NULL, NULL);
            dom->win32_require("WideCharToMultiByte(1)", n_chars != 0);
        }
        LocalFree(wargv);
#endif
        size_t vec_fill = sizeof(rust_str *) * argc;
        size_t vec_alloc = next_power_of_two(sizeof(rust_vec) + vec_fill);
        void *mem = dom->malloc(vec_alloc);
        args = new (mem) rust_vec(dom, vec_alloc, 0, NULL);
        rust_str **strs = (rust_str**) &args->data[0];
        for (int i = 0; i < argc; ++i) {
            size_t str_fill = strlen(argv[i]) + 1;
            size_t str_alloc = next_power_of_two(sizeof(rust_str) + str_fill);
            mem = dom->malloc(str_alloc);
            strs[i] = new (mem) rust_str(dom, str_alloc, str_fill,
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
                dom->free(strs[i]);
            dom->free(args);
        }

#ifdef __WIN32__
        for (int i = 0; i < argc; ++i) {
            dom->free(argv[i]);
        }
        dom->free(argv);
#endif
    }
};

/**
 * Main entry point into the Rust runtime. Here we create a Rust service,
 * initialize the kernel, create the root domain and run it.
 */

extern "C" CDECL int
rust_start(uintptr_t main_fn, int argc, char **argv, void* crate_map,
           uintptr_t main_fn_cdecl) {

    update_log_settings(crate_map, getenv("RUST_LOG"));
    rust_srv *srv = new rust_srv();
    rust_kernel *kernel = new rust_kernel(srv);
    kernel->start();
    rust_handle<rust_dom> *handle = kernel->create_domain("main");
    rust_dom *dom = handle->referent();
    command_line_args *args = new (dom) command_line_args(dom, argc, argv);

    DLOG(dom, dom, "startup: %d args in 0x%" PRIxPTR,
             args->argc, (uintptr_t)args->args);
    for (int i = 0; i < args->argc; i++) {
        DLOG(dom, dom, "startup: arg[%d] = '%s'", i, args->argv[i]);
    }

    if(main_fn) { printf("using new cdecl main\n"); }
    else { printf("using old cdecl main\n"); }
    dom->root_task->start(main_fn ? main_fn : main_fn_cdecl,
                          (uintptr_t)args->args, sizeof(args->args));

    int ret = dom->start_main_loop();
    delete args;
    kernel->destroy_domain(dom);
    kernel->join_all_domains();
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
