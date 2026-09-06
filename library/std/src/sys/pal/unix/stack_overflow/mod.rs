#![cfg_attr(test, allow(dead_code))]
#![forbid(unsafe_op_in_unsafe_fn)]

mod guard_page;

cfg_select! {
    any(
        target_os = "linux",
        target_os = "freebsd",
        target_os = "hurd",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "solaris",
        target_os = "illumos",
    ) => {
        mod thread_info;
        mod handler_signal;
        use handler_signal as handler;
    }
    target_os = "cygwin" => {
        mod handler_cygwin;
        use handler_cygwin as handler;
    }
    // This is intentionally not enabled on iOS/tvOS/watchOS/visionOS, as it uses
    // several symbols that might lead to rejections from the App Store, namely
    // `sigaction`, `sigaltstack`, `sysctlbyname`, `mmap`, `munmap` and `mprotect`.
    //
    // This might be overly cautious, though it is also what Swift does (and they
    // usually have fewer qualms about forwards compatibility, since the runtime
    // is shipped with the OS):
    // <https://github.com/apple/swift/blob/swift-5.10-RELEASE/stdlib/public/runtime/CrashHandlerMacOS.cpp>
    _ => {
        mod handler_none;
        use handler_none as handler;
    }
}

/// # Safety
/// Must be called only once, on the main thread, during program startup.
pub unsafe fn init() {
    // miri models neither signals, stack overflows nor guard pages. Also, this
    // code has some synchronization properties that we don't want to expose to
    // user code, hence we disable it on miri.
    if cfg!(miri) {
        return;
    }

    // SAFETY:
    // This is only called on the main thread, and since it is still early
    // in the programs lifetime there is (almost) certainly enough stack
    // space left to install the guard page.
    let guard_page_range = unsafe { guard_page::install_main_guard() };

    // Even for panic=immediate-abort, installing the guard pages is important
    // for soundness. That said, we do not care about giving nice stackoverflow
    // messages via our custom signal handler, just exit early and let the user
    // enjoy the segfault.
    if cfg!(panic = "immediate-abort") {
        return;
    }

    handler::init(guard_page_range);
}

pub struct Handler {
    data: *mut libc::c_void,
}

impl Handler {
    pub unsafe fn new() -> Handler {
        handler::make_handler(false)
    }

    fn null() -> Handler {
        Handler { data: crate::ptr::null_mut() }
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        unsafe {
            handler::drop_handler(self.data);
        }
    }
}
