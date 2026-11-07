//@ignore-target: windows # No libc socket on Windows
//@compile-flags: -Zmiri-disable-isolation
//@normalize-stderr-test: "address resolution failed: .*" -> "address resolution failed: $$MSG"

#[path = "../../utils/libc.rs"]
mod libc_utils;
#[path = "../../utils/mod.rs"]
mod utils;

use std::ffi::CString;
use std::ptr;

use libc_utils::*;

/// Test doing address resolution using the `getaddrinfo` syscall.
/// This also tests freeing the address linked list using `freeaddrinfo`.
fn main() {
    let node_c_str = CString::new("this-is-not-a-valid-address").unwrap();
    let service_c_str = CString::new("8080").unwrap();

    let mut hints: libc::addrinfo = unsafe { std::mem::zeroed() };
    hints.ai_socktype = libc::SOCK_STREAM;
    let mut res: *mut libc::addrinfo = ptr::null_mut();
    let retval =
        unsafe { libc::getaddrinfo(node_c_str.as_ptr(), service_c_str.as_ptr(), &hints, &mut res) };

    // We return a system error.
    assert_eq!(retval, libc::EAI_SYSTEM);
    // Last error should be a generic protocol error.
    assert_eq!(errno(), libc::EPROTO);
}
