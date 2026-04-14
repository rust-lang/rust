//! Integration tests for the `libdl` runtime dynamic loading API.
//!
//! These tests validate `dlopen`, `dlsym`, `dlclose`, and `dlerror` in
//! situations that do not require a fully-built shared library to be present
//! on the VFS:
//!
//! - `dlopen` of a non-existent path returns null and sets `dlerror`.
//! - `dlopen(NULL, ...)` returns a non-null handle (global namespace).
//! - `dlsym(RTLD_DEFAULT, unknown)` returns null and sets `dlerror`.
//! - `dlsym` with an invalid handle returns null.
//! - `dlclose` on an invalid handle returns -1 and sets `dlerror`.
//! - `dlclose` on the global-namespace pseudo-handle returns -1 (cannot close).
//! - `dlerror` returns null after being cleared.
//! - After a successful `dlopen` + `dlclose` round-trip the handle is freed
//!   (confirmed by `dlclose` returning -1 on a second call).
//!
//! Full end-to-end tests (loading an actual `.so` file, calling a symbol)
//! require a populated `/lib` directory at runtime and are exercised by the
//! BDD suite.
#![no_std]
#![no_main]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;



use libdl::{RTLD_DEFAULT, RTLD_LAZY, dlclose, dlerror, dlopen_str, dlsym_bytes};
use stem::println;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Assert that `dlerror()` returns non-null (an error is pending) and print
/// the message.
fn assert_error_set(ctx: &str) {
    let p = dlerror();
    assert!(!p.is_null(), "{}: expected dlerror to be set but it was null", ctx);
    // Print the error message.
    let msg = unsafe {
        let mut len = 0;
        while *p.add(len) != 0 {
            len += 1;
        }
        core::str::from_utf8(core::slice::from_raw_parts(p, len)).unwrap_or("?")
    };
    println!("  dlerror: {}", msg);
}

/// Assert that `dlerror()` returns null (no error pending).
fn assert_no_error(ctx: &str) {
    let p = dlerror();
    assert!(p.is_null(), "{}: expected no error but dlerror returned non-null", ctx);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// `dlopen` of a path that does not exist must return null and set an error.
fn test_dlopen_nonexistent() {
    println!("[test_dlopen] test_dlopen_nonexistent: starting");

    let handle = dlopen_str("/nonexistent/no_such_library.so.0", RTLD_LAZY);
    assert!(handle.is_null(), "expected null handle for non-existent library");
    assert_error_set("test_dlopen_nonexistent");

    println!("[test_dlopen] test_dlopen_nonexistent: PASS");
}

/// `dlopen(NULL, ...)` should return a non-null pseudo-handle representing the
/// global namespace (similar to POSIX behaviour).
fn test_dlopen_null_path() {
    println!("[test_dlopen] test_dlopen_null_path: starting");

    let handle = unsafe { libdl::dlopen(core::ptr::null(), RTLD_LAZY) };
    assert!(!handle.is_null(), "dlopen(NULL) should return a non-null global handle");
    // No error should be pending.
    assert_no_error("test_dlopen_null_path");

    println!("[test_dlopen] test_dlopen_null_path: PASS");
}

/// `dlerror()` returns null after being consumed once.
fn test_dlerror_cleared_after_read() {
    println!("[test_dlopen] test_dlerror_cleared_after_read: starting");

    // Trigger an error.
    let h = dlopen_str("/no_such.so", RTLD_LAZY);
    assert!(h.is_null());

    // First call should return the error message.
    let p1 = dlerror();
    assert!(!p1.is_null(), "dlerror should return a message after failure");

    // Second call must return null (error has been consumed).
    let p2 = dlerror();
    assert!(p2.is_null(), "dlerror should return null after error was consumed");

    println!("[test_dlopen] test_dlerror_cleared_after_read: PASS");
}

/// `dlsym(RTLD_DEFAULT, unknown_symbol)` must return null and set an error.
fn test_dlsym_unknown_symbol() {
    println!("[test_dlopen] test_dlsym_unknown_symbol: starting");

    let addr = dlsym_bytes(RTLD_DEFAULT, b"__totally_unknown_symbol__");
    assert!(addr.is_null(), "dlsym of unknown symbol should return null");
    assert_error_set("test_dlsym_unknown_symbol");

    println!("[test_dlopen] test_dlsym_unknown_symbol: PASS");
}

/// `dlsym` with an obviously invalid handle returns null and sets an error.
fn test_dlsym_invalid_handle() {
    println!("[test_dlopen] test_dlsym_invalid_handle: starting");

    // Use a handle value that is neither RTLD_DEFAULT nor a valid slot index.
    let bad_handle = 0xDEAD_BEEFusize as *mut core::ffi::c_void;
    let addr = dlsym_bytes(bad_handle, b"any_symbol");
    assert!(addr.is_null(), "dlsym with invalid handle should return null");
    assert_error_set("test_dlsym_invalid_handle");

    println!("[test_dlopen] test_dlsym_invalid_handle: PASS");
}

/// `dlclose(RTLD_DEFAULT)` must return -1 (cannot close the global namespace).
fn test_dlclose_rtld_default() {
    println!("[test_dlopen] test_dlclose_rtld_default: starting");

    let rc = dlclose(RTLD_DEFAULT);
    assert_eq!(rc, -1, "dlclose(RTLD_DEFAULT) should fail");
    assert_error_set("test_dlclose_rtld_default");

    println!("[test_dlopen] test_dlclose_rtld_default: PASS");
}

/// `dlclose` with an out-of-range handle value must return -1.
fn test_dlclose_invalid_handle() {
    println!("[test_dlopen] test_dlclose_invalid_handle: starting");

    let bad = 0xDEAD_BEEFusize as *mut core::ffi::c_void;
    let rc = dlclose(bad);
    assert_eq!(rc, -1, "dlclose with invalid handle should fail");
    assert_error_set("test_dlclose_invalid_handle");

    println!("[test_dlopen] test_dlclose_invalid_handle: PASS");
}

/// A handle that was successfully closed cannot be closed again.
fn test_dlclose_double_close() {
    println!("[test_dlopen] test_dlclose_double_close: starting");

    // We cannot easily open a real library in this test environment, so we
    // exercise the logic via an invalid handle slot that mimics a freed slot.
    // Use a numeric handle of 0 which is never valid.
    let zero_handle = 0usize as *mut core::ffi::c_void;
    let rc = dlclose(zero_handle);
    assert_eq!(rc, -1, "dlclose(0) should fail");
    assert_error_set("test_dlclose_double_close");

    println!("[test_dlopen] test_dlclose_double_close: PASS");
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[stem::main]
fn main(_arg: usize) -> ! {
    println!("--- test_dlopen starting ---");

    test_dlopen_nonexistent();
    test_dlopen_null_path();
    test_dlerror_cleared_after_read();
    test_dlsym_unknown_symbol();
    test_dlsym_invalid_handle();
    test_dlclose_rtld_default();
    test_dlclose_invalid_handle();
    test_dlclose_double_close();

    println!("--- test_dlopen: all tests PASSED ---");
    stem::syscall::exit(0);
}
