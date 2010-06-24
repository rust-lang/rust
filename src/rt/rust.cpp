#include "rust_internal.h"
#include "util/array_list.h"


// #define TRACK_ALLOCATIONS
// For debugging, keeps track of live allocations, so you can find out
// exactly what leaked.

#ifdef TRACK_ALLOCATIONS
array_list<void *> allocation_list;
#endif

rust_srv::rust_srv() :
    live_allocs(0)
{
}

rust_srv::~rust_srv()
{
    if (live_allocs != 0) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "leaked memory in rust main loop (%" PRIuPTR " objects)",
                 live_allocs);
#ifdef TRACK_ALLOCATIONS
        for (size_t i = 0; i < allocation_list.size(); i++) {
            if (allocation_list[i] != NULL) {
                printf("allocation 0x%" PRIxPTR " was not freed\n",
                        (uintptr_t) allocation_list[i]);
            }
        }
#endif
        fatal(msg, __FILE__, __LINE__);
    }
}

void
rust_srv::log(char const *str)
{
    printf("rt: %s\n", str);
}



void *
rust_srv::malloc(size_t bytes)
{
    ++live_allocs;
    void * val = ::malloc(bytes);
#ifdef TRACK_ALLOCATIONS
    allocation_list.append(val);
#endif
    return val;
}

void *
rust_srv::realloc(void *p, size_t bytes)
{
    if (!p) {
        live_allocs++;
    }
    void * val = ::realloc(p, bytes);
#ifdef TRACK_ALLOCATIONS
    if (allocation_list.replace(p, val) == NULL) {
        fatal("not in allocation_list", __FILE__, __LINE__);
    }
#endif
    return val;
}

void
rust_srv::free(void *p)
{
    if (live_allocs < 1) {
        fatal("live_allocs < 1", __FILE__, __LINE__);
    }
    live_allocs--;
    ::free(p);
#ifdef TRACK_ALLOCATIONS
    if (allocation_list.replace(p, NULL) == NULL) {
        fatal("not in allocation_list", __FILE__, __LINE__);
    }
#endif
}

void
rust_srv::fatal(char const *expr, char const *file, size_t line)
{
    char buf[1024];
    snprintf(buf, sizeof(buf),
             "fatal, '%s' failed, %s:%d",
             expr, file, (int)line);
    log(buf);
    exit(1);
}

rust_srv *
rust_srv::clone()
{
    return new rust_srv();
}


int
rust_main_loop(rust_dom *dom)
{
    // Make sure someone is watching, to pull us out of infinite loops.
    rust_timer timer(*dom);

    int rval;
    rust_task *task;

    dom->log(rust_log::DOM,
            "running main-loop on domain 0x%" PRIxPTR, dom);
    dom->logptr("exit-task glue",
            dom->root_crate->get_exit_task_glue());

    while ((task = dom->sched()) != NULL) {
        I(dom, task->running());

        dom->log(rust_log::TASK,
                "activating task 0x%" PRIxPTR ", sp=0x%" PRIxPTR,
                (uintptr_t)task, task->rust_sp);

        dom->interrupt_flag = 0;

        dom->activate(task);

        dom->log(rust_log::TASK,
                 "returned from task 0x%" PRIxPTR
                 " in state '%s', sp=0x%" PRIxPTR,
                 (uintptr_t)task,
                 dom->state_vec_name(task->state),
                 task->rust_sp);

        I(dom, task->rust_sp >= (uintptr_t) &task->stk->data[0]);
        I(dom, task->rust_sp < task->stk->limit);

        dom->reap_dead_tasks();
    }

    dom->log(rust_log::DOM, "finished main-loop (dom.rval = %d)", dom->rval);
    rval = dom->rval;

    return rval;
}


struct
command_line_args
{
    rust_dom &dom;
    int argc;
    char **argv;

    // vec[str] passed to rust_task::start.
    rust_vec *args;

    command_line_args(rust_dom &dom,
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
        dom.win32_require("CommandLineToArgvW", argv != NULL);
        argv = (char **) dom.malloc(sizeof(char*) * argc);
        for (int i = 0; i < argc; ++i) {
            int n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                              NULL, 0, NULL, NULL);
            dom.win32_require("WideCharToMultiByte(0)", n_chars != 0);
            argv[i] = (char *) dom.malloc(n_chars);
            n_chars = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                          argv[i], n_chars, NULL, NULL);
            dom.win32_require("WideCharToMultiByte(1)", n_chars != 0);
        }
        LocalFree(wargv);
#endif
        size_t vec_fill = sizeof(rust_str *) * argc;
        size_t vec_alloc = next_power_of_two(sizeof(rust_vec) + vec_fill);
        void *mem = dom.malloc(vec_alloc);
        args = new (mem) rust_vec(&dom, vec_alloc, 0, NULL);
        rust_str **strs = (rust_str**) &args->data[0];
        for (int i = 0; i < argc; ++i) {
            size_t str_fill = strlen(argv[i]) + 1;
            size_t str_alloc = next_power_of_two(sizeof(rust_str) + str_fill);
            mem = dom.malloc(str_alloc);
            strs[i] = new (mem) rust_str(&dom, str_alloc, str_fill,
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
                dom.free(strs[i]);
            dom.free(args);
        }

#ifdef __WIN32__
        for (int i = 0; i < argc; ++i) {
            dom.free(argv[i]);
        }
        dom.free(argv);
#endif
    }
};


extern "C" CDECL int
rust_start(uintptr_t main_fn, rust_crate const *crate, int argc, char **argv)
{
    int ret;
    {
        rust_srv srv;
        rust_dom dom(&srv, crate);
        command_line_args args(dom, argc, argv);

        dom.log(rust_log::DOM, "startup: %d args", args.argc);
        for (int i = 0; i < args.argc; ++i)
            dom.log(rust_log::DOM,
                    "startup: arg[%d] = '%s'", i, args.argv[i]);

        if (dom._log.is_tracing(rust_log::DWARF)) {
            rust_crate_reader rdr(&dom, crate);
        }

        uintptr_t main_args[3] = { 0, 0, (uintptr_t)args.args };

        dom.root_task->start(crate->get_exit_task_glue(),
                             main_fn,
                             (uintptr_t)&main_args,
                             sizeof(main_args));

        ret = rust_main_loop(&dom);
    }

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
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
