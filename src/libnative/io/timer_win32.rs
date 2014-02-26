// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Timers based on win32 WaitableTimers
//!
//! This implementation is meant to be used solely on windows. As with other
//! implementations, there is a worker thread which is doing all the waiting on
//! a large number of timers for all active timers in the system. This worker
//! thread uses the select() equivalent, WaitForMultipleObjects. One of the
//! objects being waited on is a signal into the worker thread to notify that
//! the incoming channel should be looked at.
//!
//! Other than that, the implementation is pretty straightforward in terms of
//! the other two implementations of timers with nothing *that* new showing up.

use std::comm::Data;
use libc;
use std::ptr;
use std::rt::rtio;

use io::timer_helper;
use io::IoResult;

pub struct Timer {
    priv obj: libc::HANDLE,
    priv on_worker: bool,
}

pub enum Req {
    NewTimer(libc::HANDLE, Sender<()>, bool),
    RemoveTimer(libc::HANDLE, Sender<()>),
    Shutdown,
}

fn helper(input: libc::HANDLE, messages: Receiver<Req>) {
    let mut objs = ~[input];
    let mut chans = ~[];

    'outer: loop {
        let idx = unsafe {
            imp::WaitForMultipleObjects(objs.len() as libc::DWORD,
                                        objs.as_ptr(),
                                        0 as libc::BOOL,
                                        libc::INFINITE)
        };

        if idx == 0 {
            loop {
                match messages.try_recv() {
                    Data(NewTimer(obj, c, one)) => {
                        objs.push(obj);
                        chans.push((c, one));
                    }
                    Data(RemoveTimer(obj, c)) => {
                        c.send(());
                        match objs.iter().position(|&o| o == obj) {
                            Some(i) => {
                                drop(objs.remove(i));
                                drop(chans.remove(i - 1));
                            }
                            None => {}
                        }
                    }
                    Data(Shutdown) => {
                        assert_eq!(objs.len(), 1);
                        assert_eq!(chans.len(), 0);
                        break 'outer;
                    }
                    _ => break
                }
            }
        } else {
            let remove = {
                match &chans[idx - 1] {
                    &(ref c, oneshot) => !c.try_send(()) || oneshot
                }
            };
            if remove {
                drop(objs.remove(idx as uint));
                drop(chans.remove(idx as uint - 1));
            }
        }
    }
}

impl Timer {
    pub fn new() -> IoResult<Timer> {
        timer_helper::boot(helper);

        let obj = unsafe {
            imp::CreateWaitableTimerA(ptr::mut_null(), 0, ptr::null())
        };
        if obj.is_null() {
            Err(super::last_error())
        } else {
            Ok(Timer { obj: obj, on_worker: false, })
        }
    }

    pub fn sleep(ms: u64) {
        use std::rt::rtio::RtioTimer;
        let mut t = Timer::new().ok().expect("must allocate a timer!");
        t.sleep(ms);
    }

    fn remove(&mut self) {
        if !self.on_worker { return }

        let (tx, rx) = channel();
        timer_helper::send(RemoveTimer(self.obj, tx));
        rx.recv();

        self.on_worker = false;
    }
}

impl rtio::RtioTimer for Timer {
    fn sleep(&mut self, msecs: u64) {
        self.remove();

        // there are 10^6 nanoseconds in a millisecond, and the parameter is in
        // 100ns intervals, so we multiply by 10^4.
        let due = -(msecs * 10000) as libc::LARGE_INTEGER;
        assert_eq!(unsafe {
            imp::SetWaitableTimer(self.obj, &due, 0, ptr::null(),
                                  ptr::mut_null(), 0)
        }, 1);

        let _ = unsafe { imp::WaitForSingleObject(self.obj, libc::INFINITE) };
    }

    fn oneshot(&mut self, msecs: u64) -> Receiver<()> {
        self.remove();
        let (tx, rx) = channel();

        // see above for the calculation
        let due = -(msecs * 10000) as libc::LARGE_INTEGER;
        assert_eq!(unsafe {
            imp::SetWaitableTimer(self.obj, &due, 0, ptr::null(),
                                  ptr::mut_null(), 0)
        }, 1);

        timer_helper::send(NewTimer(self.obj, tx, true));
        self.on_worker = true;
        return rx;
    }

    fn period(&mut self, msecs: u64) -> Receiver<()> {
        self.remove();
        let (tx, rx) = channel();

        // see above for the calculation
        let due = -(msecs * 10000) as libc::LARGE_INTEGER;
        assert_eq!(unsafe {
            imp::SetWaitableTimer(self.obj, &due, msecs as libc::LONG,
                                  ptr::null(), ptr::mut_null(), 0)
        }, 1);

        timer_helper::send(NewTimer(self.obj, tx, false));
        self.on_worker = true;

        return rx;
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        self.remove();
        assert!(unsafe { libc::CloseHandle(self.obj) != 0 });
    }
}

mod imp {
    use libc::{LPSECURITY_ATTRIBUTES, BOOL, LPCSTR, HANDLE, LARGE_INTEGER,
                    LONG, LPVOID, DWORD, c_void};

    pub type PTIMERAPCROUTINE = *c_void;

    extern "system" {
        pub fn CreateWaitableTimerA(lpTimerAttributes: LPSECURITY_ATTRIBUTES,
                                    bManualReset: BOOL,
                                    lpTimerName: LPCSTR) -> HANDLE;
        pub fn SetWaitableTimer(hTimer: HANDLE,
                                pDueTime: *LARGE_INTEGER,
                                lPeriod: LONG,
                                pfnCompletionRoutine: PTIMERAPCROUTINE,
                                lpArgToCompletionRoutine: LPVOID,
                                fResume: BOOL) -> BOOL;
        pub fn WaitForMultipleObjects(nCount: DWORD,
                                      lpHandles: *HANDLE,
                                      bWaitAll: BOOL,
                                      dwMilliseconds: DWORD) -> DWORD;
        pub fn WaitForSingleObject(hHandle: HANDLE,
                                   dwMilliseconds: DWORD) -> DWORD;
    }
}
