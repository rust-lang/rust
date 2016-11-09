// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ffi::OsStr;
use std::io;
use std::os::raw;
use std::os::windows::prelude::*;

pub struct RegistryKey(Repr);

type HKEY = *mut u8;
type DWORD = u32;
type LPDWORD = *mut DWORD;
type LPCWSTR = *const u16;
type LPWSTR = *mut u16;
type LONG = raw::c_long;
type PHKEY = *mut HKEY;
type PFILETIME = *mut u8;
type LPBYTE = *mut u8;
type REGSAM = u32;

const ERROR_SUCCESS: DWORD = 0;
const HKEY_LOCAL_MACHINE: HKEY = 0x80000002 as HKEY;
const KEY_READ: DWORD = 0x20019;
const KEY_WOW64_32KEY: DWORD = 0x200;

#[link(name = "advapi32")]
extern "system" {
    fn RegOpenKeyExW(key: HKEY,
                     lpSubKey: LPCWSTR,
                     ulOptions: DWORD,
                     samDesired: REGSAM,
                     phkResult: PHKEY) -> LONG;
    fn RegCloseKey(hKey: HKEY) -> LONG;
}

struct OwnedKey(HKEY);

enum Repr {
    Const(HKEY),
    Owned(OwnedKey),
}

unsafe impl Sync for Repr {}
unsafe impl Send for Repr {}

pub static LOCAL_MACHINE: RegistryKey =
    RegistryKey(Repr::Const(HKEY_LOCAL_MACHINE));

impl RegistryKey {
    fn raw(&self) -> HKEY {
        match self.0 {
            Repr::Const(val) => val,
            Repr::Owned(ref val) => val.0,
        }
    }

    pub fn open(&self, key: &OsStr) -> io::Result<RegistryKey> {
        let key = key.encode_wide().chain(Some(0)).collect::<Vec<_>>();
        let mut ret = 0 as *mut _;
        let err = unsafe {
            RegOpenKeyExW(self.raw(), key.as_ptr(), 0,
                          KEY_READ | KEY_WOW64_32KEY, &mut ret)
        };
        if err == ERROR_SUCCESS as LONG {
            Ok(RegistryKey(Repr::Owned(OwnedKey(ret))))
        } else {
            Err(io::Error::from_raw_os_error(err as i32))
        }
    }
}

impl Drop for OwnedKey {
    fn drop(&mut self) {
        unsafe { RegCloseKey(self.0); }
    }
}
