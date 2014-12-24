// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{BOOL, DWORD, c_void, LPVOID};
use libc::types::os::arch::extra::BOOLEAN;

pub type LPCRITICAL_SECTION = *mut c_void;
pub type LPCONDITION_VARIABLE = *mut CONDITION_VARIABLE;
pub type LPSRWLOCK = *mut SRWLOCK;

#[cfg(target_arch = "x86")]
pub const CRITICAL_SECTION_SIZE: uint = 24;
#[cfg(target_arch = "x86_64")]
pub const CRITICAL_SECTION_SIZE: uint = 40;

#[repr(C)]
pub struct CONDITION_VARIABLE { pub ptr: LPVOID }
#[repr(C)]
pub struct SRWLOCK { pub ptr: LPVOID }

pub const CONDITION_VARIABLE_INIT: CONDITION_VARIABLE = CONDITION_VARIABLE {
    ptr: 0 as *mut _,
};
pub const SRWLOCK_INIT: SRWLOCK = SRWLOCK { ptr: 0 as *mut _ };

extern "system" {
    // critical sections
    pub fn InitializeCriticalSectionAndSpinCount(
                    lpCriticalSection: LPCRITICAL_SECTION,
                    dwSpinCount: DWORD) -> BOOL;
    pub fn DeleteCriticalSection(lpCriticalSection: LPCRITICAL_SECTION);
    pub fn EnterCriticalSection(lpCriticalSection: LPCRITICAL_SECTION);
    pub fn LeaveCriticalSection(lpCriticalSection: LPCRITICAL_SECTION);
    pub fn TryEnterCriticalSection(lpCriticalSection: LPCRITICAL_SECTION) -> BOOL;

    // condition variables
    pub fn SleepConditionVariableCS(ConditionVariable: LPCONDITION_VARIABLE,
                                    CriticalSection: LPCRITICAL_SECTION,
                                    dwMilliseconds: DWORD) -> BOOL;
    pub fn WakeConditionVariable(ConditionVariable: LPCONDITION_VARIABLE);
    pub fn WakeAllConditionVariable(ConditionVariable: LPCONDITION_VARIABLE);

    // slim rwlocks
    pub fn AcquireSRWLockExclusive(SRWLock: LPSRWLOCK);
    pub fn AcquireSRWLockShared(SRWLock: LPSRWLOCK);
    pub fn ReleaseSRWLockExclusive(SRWLock: LPSRWLOCK);
    pub fn ReleaseSRWLockShared(SRWLock: LPSRWLOCK);
    pub fn TryAcquireSRWLockExclusive(SRWLock: LPSRWLOCK) -> BOOLEAN;
    pub fn TryAcquireSRWLockShared(SRWLock: LPSRWLOCK) -> BOOLEAN;
}

