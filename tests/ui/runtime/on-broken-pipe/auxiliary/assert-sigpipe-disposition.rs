// It is UB to unwind out of `fn start()` according to
// https://doc.rust-lang.org/beta/unstable-book/language-features/start.html so
// panic with abort to avoid UB:
//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic so panic=abort works

#![feature(rustc_private)]
#![no_main]

extern crate libc;

// Use no_main so we don't have a runtime that messes with SIGPIPE.
#[no_mangle]
extern "C" fn main(argc: core::ffi::c_int, argv: *const *const u8) -> core::ffi::c_int {
    assert_eq!(argc, 2, "Must pass SIG_IGN or SIG_DFL as first arg");
    let arg1 = unsafe { core::ffi::CStr::from_ptr(*argv.offset(1) as *const libc::c_char) }
        .to_str()
        .unwrap();

    let expected = match arg1 {
        "SIG_IGN" => libc::SIG_IGN,
        "SIG_DFL" => libc::SIG_DFL,
        arg => panic!("Must pass SIG_IGN or SIG_DFL as first arg. Got: {}", arg),
    };

    let actual = unsafe {
        let mut actual: libc::sigaction = core::mem::zeroed();
        libc::sigaction(libc::SIGPIPE, core::ptr::null(), &mut actual);
        actual.sa_sigaction
    };

    assert_eq!(actual, expected, "actual and expected SIGPIPE disposition in child differs");

    0
}
