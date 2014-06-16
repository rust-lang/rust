// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

#[cfg(unix)] use libc::c_int;
#[cfg(unix)] use core::ptr::null;
#[cfg(windows)] use libc::types::os::arch::extra::{DWORD, LPVOID, BOOL};

#[cfg(unix)]
pub type Key = pthread_key_t;

#[cfg(unix)]
pub unsafe fn create(key: &mut Key) {
    assert!(pthread_key_create(key, null()) == 0);
}

#[cfg(unix)]
pub unsafe fn set(key: Key, value: *mut u8) {
    assert!(pthread_setspecific(key, value) == 0);
}

#[cfg(unix)]
pub unsafe fn get(key: Key) -> *mut u8 {
    pthread_getspecific(key)
}

#[cfg(unix)]
pub unsafe fn destroy(key: Key) {
    assert!(pthread_key_delete(key) == 0);
}

#[cfg(target_os = "macos")]
#[allow(non_camel_case_types)] // foreign type
type pthread_key_t = ::libc::c_ulong;

#[cfg(target_os="linux")]
#[cfg(target_os="freebsd")]
#[cfg(target_os="android")]
#[cfg(target_os = "ios")]
#[allow(non_camel_case_types)] // foreign type
type pthread_key_t = ::libc::c_uint;

#[cfg(unix)]
extern {
    fn pthread_key_create(key: *mut pthread_key_t, dtor: *u8) -> c_int;
    fn pthread_key_delete(key: pthread_key_t) -> c_int;
    fn pthread_getspecific(key: pthread_key_t) -> *mut u8;
    fn pthread_setspecific(key: pthread_key_t, value: *mut u8) -> c_int;
}

#[cfg(windows)]
pub type Key = DWORD;

#[cfg(windows)]
pub unsafe fn create(key: &mut Key) {
    static TLS_OUT_OF_INDEXES: DWORD = 0xFFFFFFFF;
    *key = TlsAlloc();
    assert!(*key != TLS_OUT_OF_INDEXES);
}

#[cfg(windows)]
pub unsafe fn set(key: Key, value: *mut u8) {
    assert!(0 != TlsSetValue(key, value as *mut ::libc::c_void))
}

#[cfg(windows)]
pub unsafe fn get(key: Key) -> *mut u8 {
    TlsGetValue(key) as *mut u8
}

#[cfg(windows)]
pub unsafe fn destroy(key: Key) {
    assert!(TlsFree(key) != 0);
}

#[cfg(windows)]
#[allow(non_snake_case_functions)]
extern "system" {
    fn TlsAlloc() -> DWORD;
    fn TlsFree(dwTlsIndex: DWORD) -> BOOL;
    fn TlsGetValue(dwTlsIndex: DWORD) -> LPVOID;
    fn TlsSetValue(dwTlsIndex: DWORD, lpTlsvalue: LPVOID) -> BOOL;
}

#[cfg(test)]
mod test {
    use std::prelude::*;
    use super::*;

    #[test]
    fn tls_smoke_test() {
        use std::mem::transmute;
        unsafe {
            let mut key = 0;
            let value = box 20;
            create(&mut key);
            set(key, transmute(value));
            let value: Box<int> = transmute(get(key));
            assert_eq!(value, box 20);
            let value = box 30;
            set(key, transmute(value));
            let value: Box<int> = transmute(get(key));
            assert_eq!(value, box 30);
        }
    }
}
