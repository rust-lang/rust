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
        let actual = unsafe {
            let mut actual: libc::sigaction = std::mem::zeroed();
            libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut actual);
            actual.sa_sigaction
        };

        let expected = match expected_handler {
            SignalHandler::Ignore => libc::SIG_IGN,
            SignalHandler::Default => libc::SIG_DFL,
        };

        assert_eq!(actual, expected, "actual and expected SIGPIPE disposition differs");
    }
}
