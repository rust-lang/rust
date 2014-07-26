// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the helper thread for the timer module
//!
//! This module contains the management necessary for the timer worker thread.
//! This thread is responsible for performing the send()s on channels for timers
//! that are using channels instead of a blocking call.
//!
//! The timer thread is lazily initialized, and it's shut down via the
//! `shutdown` function provided. It must be maintained as an invariant that
//! `shutdown` is only called when the entire program is finished. No new timers
//! can be created in the future and there must be no active timers at that
//! time.

#![macro_escape]

use std::mem;
use std::rt::bookkeeping;
use std::rt::mutex::StaticNativeMutex;
use std::rt;
use std::cell::UnsafeCell;

use task;

/// A structure for management of a helper thread.
///
/// This is generally a static structure which tracks the lifetime of a helper
/// thread.
///
/// The fields of this helper are all public, but they should not be used, this
/// is for static initialization.
pub struct Helper<M> {
    /// Internal lock which protects the remaining fields
    pub lock: StaticNativeMutex,

    // You'll notice that the remaining fields are UnsafeCell<T>, and this is
    // because all helper thread operations are done through &self, but we need
    // these to be mutable (once `lock` is held).

    /// Lazily allocated channel to send messages to the helper thread.
    pub chan: UnsafeCell<*mut Sender<M>>,

    /// OS handle used to wake up a blocked helper thread
    pub signal: UnsafeCell<uint>,

    /// Flag if this helper thread has booted and been initialized yet.
    pub initialized: UnsafeCell<bool>,
}

macro_rules! helper_init( (static mut $name:ident: Helper<$m:ty>) => (
    static mut $name: Helper<$m> = Helper {
        lock: ::std::rt::mutex::NATIVE_MUTEX_INIT,
        chan: ::std::cell::UnsafeCell { value: 0 as *mut Sender<$m> },
        signal: ::std::cell::UnsafeCell { value: 0 },
        initialized: ::std::cell::UnsafeCell { value: false },
    };
) )

impl<M: Send> Helper<M> {
    /// Lazily boots a helper thread, becoming a no-op if the helper has already
    /// been spawned.
    ///
    /// This function will check to see if the thread has been initialized, and
    /// if it has it returns quickly. If initialization has not happened yet,
    /// the closure `f` will be run (inside of the initialization lock) and
    /// passed to the helper thread in a separate task.
    ///
    /// This function is safe to be called many times.
    pub fn boot<T: Send>(&'static self,
                         f: || -> T,
                         helper: fn(imp::signal, Receiver<M>, T)) {
        unsafe {
            let _guard = self.lock.lock();
            if !*self.initialized.get() {
                let (tx, rx) = channel();
                *self.chan.get() = mem::transmute(box tx);
                let (receive, send) = imp::new();
                *self.signal.get() = send as uint;

                let t = f();
                task::spawn(proc() {
                    bookkeeping::decrement();
                    helper(receive, rx, t);
                    self.lock.lock().signal()
                });

                rt::at_exit(proc() { self.shutdown() });
                *self.initialized.get() = true;
            }
        }
    }

    /// Sends a message to a spawned worker thread.
    ///
    /// This is only valid if the worker thread has previously booted
    pub fn send(&'static self, msg: M) {
        unsafe {
            let _guard = self.lock.lock();

            // Must send and *then* signal to ensure that the child receives the
            // message. Otherwise it could wake up and go to sleep before we
            // send the message.
            assert!(!self.chan.get().is_null());
            (**self.chan.get()).send(msg);
            imp::signal(*self.signal.get() as imp::signal);
        }
    }

    fn shutdown(&'static self) {
        unsafe {
            // Shut down, but make sure this is done inside our lock to ensure
            // that we'll always receive the exit signal when the thread
            // returns.
            let guard = self.lock.lock();

            // Close the channel by destroying it
            let chan: Box<Sender<M>> = mem::transmute(*self.chan.get());
            *self.chan.get() = 0 as *mut Sender<M>;
            drop(chan);
            imp::signal(*self.signal.get() as imp::signal);

            // Wait for the child to exit
            guard.wait();
            drop(guard);

            // Clean up after ourselves
            self.lock.destroy();
            imp::close(*self.signal.get() as imp::signal);
            *self.signal.get() = 0;
        }
    }
}

#[cfg(unix)]
mod imp {
    use libc;
    use std::os;

    use io::file::FileDesc;

    pub type signal = libc::c_int;

    pub fn new() -> (signal, signal) {
        let os::Pipe { reader, writer } = unsafe { os::pipe().unwrap() };
        (reader, writer)
    }

    pub fn signal(fd: libc::c_int) {
        FileDesc::new(fd, false).inner_write([0]).ok().unwrap();
    }

    pub fn close(fd: libc::c_int) {
        let _fd = FileDesc::new(fd, true);
    }
}

#[cfg(windows)]
mod imp {
    use libc::{BOOL, LPCSTR, HANDLE, LPSECURITY_ATTRIBUTES, CloseHandle};
    use std::ptr;
    use libc;

    pub type signal = HANDLE;

    pub fn new() -> (HANDLE, HANDLE) {
        unsafe {
            let handle = CreateEventA(ptr::mut_null(), libc::FALSE, libc::FALSE,
                                      ptr::null());
            (handle, handle)
        }
    }

    pub fn signal(handle: HANDLE) {
        assert!(unsafe { SetEvent(handle) != 0 });
    }

    pub fn close(handle: HANDLE) {
        assert!(unsafe { CloseHandle(handle) != 0 });
    }

    extern "system" {
        fn CreateEventA(lpSecurityAttributes: LPSECURITY_ATTRIBUTES,
                        bManualReset: BOOL,
                        bInitialState: BOOL,
                        lpName: LPCSTR) -> HANDLE;
        fn SetEvent(hEvent: HANDLE) -> BOOL;
    }
}
