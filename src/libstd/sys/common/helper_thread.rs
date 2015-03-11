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

use prelude::v1::*;

use boxed;
use cell::UnsafeCell;
use rt;
use sync::{StaticMutex, StaticCondvar};
use sync::mpsc::{channel, Sender, Receiver};
use sys::helper_signal;

use thread;

/// A structure for management of a helper thread.
///
/// This is generally a static structure which tracks the lifetime of a helper
/// thread.
///
/// The fields of this helper are all public, but they should not be used, this
/// is for static initialization.
pub struct Helper<M> {
    /// Internal lock which protects the remaining fields
    pub lock: StaticMutex,
    pub cond: StaticCondvar,

    // You'll notice that the remaining fields are UnsafeCell<T>, and this is
    // because all helper thread operations are done through &self, but we need
    // these to be mutable (once `lock` is held).

    /// Lazily allocated channel to send messages to the helper thread.
    pub chan: UnsafeCell<*mut Sender<M>>,

    /// OS handle used to wake up a blocked helper thread
    pub signal: UnsafeCell<uint>,

    /// Flag if this helper thread has booted and been initialized yet.
    pub initialized: UnsafeCell<bool>,

    /// Flag if this helper thread has shut down
    pub shutdown: UnsafeCell<bool>,
}

unsafe impl<M:Send> Send for Helper<M> { }

unsafe impl<M:Send> Sync for Helper<M> { }

struct RaceBox(helper_signal::signal);

unsafe impl Send for RaceBox {}
unsafe impl Sync for RaceBox {}

macro_rules! helper_init { (static $name:ident: Helper<$m:ty>) => (
    static $name: Helper<$m> = Helper {
        lock: ::sync::MUTEX_INIT,
        cond: ::sync::CONDVAR_INIT,
        chan: ::cell::UnsafeCell { value: 0 as *mut Sender<$m> },
        signal: ::cell::UnsafeCell { value: 0 },
        initialized: ::cell::UnsafeCell { value: false },
        shutdown: ::cell::UnsafeCell { value: false },
    };
) }

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
    pub fn boot<T, F>(&'static self, f: F, helper: fn(helper_signal::signal, Receiver<M>, T)) where
        T: Send + 'static,
        F: FnOnce() -> T,
    {
        unsafe {
            let _guard = self.lock.lock().unwrap();
            if *self.chan.get() as uint == 0 {
                let (tx, rx) = channel();
                *self.chan.get() = boxed::into_raw(box tx);
                let (receive, send) = helper_signal::new();
                *self.signal.get() = send as uint;

                let receive = RaceBox(receive);

                let t = f();
                thread::spawn(move || {
                    helper(receive.0, rx, t);
                    let _g = self.lock.lock().unwrap();
                    *self.shutdown.get() = true;
                    self.cond.notify_one()
                });

                rt::at_exit(move || { self.shutdown() });
                *self.initialized.get() = true;
            } else if *self.chan.get() as uint == 1 {
                panic!("cannot continue usage after shutdown");
            }
        }
    }

    /// Sends a message to a spawned worker thread.
    ///
    /// This is only valid if the worker thread has previously booted
    pub fn send(&'static self, msg: M) {
        unsafe {
            let _guard = self.lock.lock().unwrap();

            // Must send and *then* signal to ensure that the child receives the
            // message. Otherwise it could wake up and go to sleep before we
            // send the message.
            assert!(*self.chan.get() as uint != 0);
            assert!(*self.chan.get() as uint != 1,
                    "cannot continue usage after shutdown");
            (**self.chan.get()).send(msg).unwrap();
            helper_signal::signal(*self.signal.get() as helper_signal::signal);
        }
    }

    fn shutdown(&'static self) {
        unsafe {
            // Shut down, but make sure this is done inside our lock to ensure
            // that we'll always receive the exit signal when the thread
            // returns.
            let mut guard = self.lock.lock().unwrap();

            let ptr = *self.chan.get();
            if ptr as uint == 1 {
                panic!("cannot continue usage after shutdown");
            }
            // Close the channel by destroying it
            let chan = Box::from_raw(*self.chan.get());
            *self.chan.get() = 1 as *mut Sender<M>;
            drop(chan);
            helper_signal::signal(*self.signal.get() as helper_signal::signal);

            // Wait for the child to exit
            while !*self.shutdown.get() {
                guard = self.cond.wait(guard).unwrap();
            }
            drop(guard);

            // Clean up after ourselves
            self.lock.destroy();
            helper_signal::close(*self.signal.get() as helper_signal::signal);
            *self.signal.get() = 0;
        }
    }
}
