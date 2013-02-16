// xfail-test not a test

type raw_thread = libc::c_void;

struct Thread {
    main: ~fn(),
    raw_thread: *raw_thread
}

impl Thread {
    static fn start(main: ~fn()) -> Thread {
        fn substart(main: &fn()) -> *raw_thread {
            unsafe { rust_raw_thread_start(main) }
        }
        let raw = substart(main);
        Thread {
            main: main,
            raw_thread: raw
        }
    }
}

impl Thread: Drop {
    fn finalize(&self) {
        unsafe { rust_raw_thread_join_delete(self.raw_thread) }
    }
}

extern {
    pub unsafe fn rust_raw_thread_start(f: &fn()) -> *raw_thread;
    pub unsafe fn rust_raw_thread_join_delete(thread: *raw_thread);
}
