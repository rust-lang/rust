// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::{BOOL, DWORD, LPVOID, LONG, HANDLE, c_ulong};
use libc::types::os::arch::extra::BOOLEAN;

pub type PCONDITION_VARIABLE = *mut CONDITION_VARIABLE;
pub type PSRWLOCK = *mut SRWLOCK;
pub type ULONG = c_ulong;
pub type ULONG_PTR = c_ulong;

#[repr(C)]
pub struct CONDITION_VARIABLE { pub ptr: LPVOID }
#[repr(C)]
pub struct SRWLOCK { pub ptr: LPVOID }
#[repr(C)]
pub struct CRITICAL_SECTION {
    CriticalSectionDebug: LPVOID,
    LockCount: LONG,
    RecursionCount: LONG,
    OwningThread: HANDLE,
    LockSemaphore: HANDLE,
    SpinCount: ULONG_PTR
}

pub const CONDITION_VARIABLE_INIT: CONDITION_VARIABLE = CONDITION_VARIABLE {
    ptr: 0 as *mut _,
};
pub const SRWLOCK_INIT: SRWLOCK = SRWLOCK { ptr: 0 as *mut _ };

extern "system" {
    // condition variables
    pub fn SleepConditionVariableSRW(ConditionVariable: PCONDITION_VARIABLE,
                                     SRWLock: PSRWLOCK,
                                     dwMilliseconds: DWORD,
                                     Flags: ULONG) -> BOOL;
    pub fn WakeConditionVariable(ConditionVariable: PCONDITION_VARIABLE);
    pub fn WakeAllConditionVariable(ConditionVariable: PCONDITION_VARIABLE);

    // slim rwlocks
    pub fn AcquireSRWLockExclusive(SRWLock: PSRWLOCK);
    pub fn AcquireSRWLockShared(SRWLock: PSRWLOCK);
    pub fn ReleaseSRWLockExclusive(SRWLock: PSRWLOCK);
    pub fn ReleaseSRWLockShared(SRWLock: PSRWLOCK);
    pub fn TryAcquireSRWLockExclusive(SRWLock: PSRWLOCK) -> BOOLEAN;
    pub fn TryAcquireSRWLockShared(SRWLock: PSRWLOCK) -> BOOLEAN;

    pub fn InitializeCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn EnterCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn TryEnterCriticalSection(CriticalSection: *mut CRITICAL_SECTION) -> BOOLEAN;
    pub fn LeaveCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
    pub fn DeleteCriticalSection(CriticalSection: *mut CRITICAL_SECTION);
}
