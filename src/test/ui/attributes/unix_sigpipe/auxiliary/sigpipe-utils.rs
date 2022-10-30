#![feature(rustc_private)]
extern crate libc;

/// So tests don't have to bring libc in scope themselves
pub enum SignalHandler {
    Ignore,
    Default,
}

/// Helper to assert that [`libc::SIGPIPE`] has the expected signal handler.
pub fn assert_sigpipe_handler(expected_handler: SignalHandler) {
    #[cfg(unix)]
    #[cfg(not(any(
        target_os = "emscripten",
        target_os = "fuchsia",
        target_os = "horizon",
        target_os = "android",
    )))]
    {
        let prev = unsafe { libc::signal(libc::SIGPIPE, libc::SIG_IGN) };

        let expected = match expected_handler {
            SignalHandler::Ignore => libc::SIG_IGN,
            SignalHandler::Default => libc::SIG_DFL,
        };
        assert_eq!(prev, expected);

        // Unlikely to matter, but restore the old value anyway
        unsafe { libc::signal(libc::SIGPIPE, prev); };
    }
}
