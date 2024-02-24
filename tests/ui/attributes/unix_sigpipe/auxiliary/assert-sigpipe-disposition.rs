// It is UB to unwind out of `fn start()` according to
// https://doc.rust-lang.org/beta/unstable-book/language-features/start.html so
// panic with abort to avoid UB:
//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic so panic=abort works

#![feature(start, rustc_private)]

extern crate libc;

// Use #[start] so we don't have a runtime that messes with SIGPIPE.
#[start]
fn start(argc: isize, argv: *const *const u8) -> isize {
    assert_eq!(argc, 2, "Must pass SIG_IGN or SIG_DFL as first arg");
    let arg1 = unsafe { std::ffi::CStr::from_ptr(*argv.offset(1) as *const libc::c_char) }
        .to_str()
        .unwrap();

    let expected = match arg1 {
        "SIG_IGN" => libc::SIG_IGN,
        "SIG_DFL" => libc::SIG_DFL,
        arg => panic!("Must pass SIG_IGN or SIG_DFL as first arg. Got: {}", arg),
    };

    let actual = unsafe {
        let mut actual: libc::sigaction = std::mem::zeroed();
        libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut actual);
        actual.sa_sigaction
    };

    assert_eq!(actual, expected, "actual and expected SIGPIPE disposition in child differs");

    0
}
