// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types, non_snake_case)]

use libc::c_void;
use std::mem;

type DWORD = u32;
type WORD = u16;
type LPVOID = *mut c_void;
type DWORD_PTR = usize;

const PROCESSOR_ARCHITECTURE_INTEL: WORD = 0;
const PROCESSOR_ARCHITECTURE_AMD64: WORD = 9;

#[repr(C)]
struct SYSTEM_INFO {
    wProcessorArchitecture: WORD,
    _wReserved: WORD,
    _dwPageSize: DWORD,
    _lpMinimumApplicationAddress: LPVOID,
    _lpMaximumApplicationAddress: LPVOID,
    _dwActiveProcessorMask: DWORD_PTR,
    _dwNumberOfProcessors: DWORD,
    _dwProcessorType: DWORD,
    _dwAllocationGranularity: DWORD,
    _wProcessorLevel: WORD,
    _wProcessorRevision: WORD,
}

extern "system" {
    fn GetNativeSystemInfo(lpSystemInfo: *mut SYSTEM_INFO);
}

pub enum Arch {
    X86,
    Amd64,
}

pub fn host_arch() -> Option<Arch> {
    let mut info = unsafe { mem::zeroed() };
    unsafe { GetNativeSystemInfo(&mut info) };
    match info.wProcessorArchitecture {
        PROCESSOR_ARCHITECTURE_INTEL => Some(Arch::X86),
        PROCESSOR_ARCHITECTURE_AMD64 => Some(Arch::Amd64),
        _ => None,
    }
}
