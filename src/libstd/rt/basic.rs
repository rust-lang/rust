// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This is a basic event loop implementation not meant for any "real purposes"
//! other than testing the scheduler and proving that it's possible to have a
//! pluggable event loop.

use prelude::*;

use cast;
use rt::rtio::{EventLoop, IoFactory, RemoteCallback, PausibleIdleCallback,
               Callback};
use unstable::sync::Exclusive;
use util;

/// This is the only exported function from this module.
pub fn event_loop() -> ~EventLoop {
    ~BasicLoop::new() as ~EventLoop
}

struct BasicLoop {
    work: ~[proc()],                  // pending work
    idle: Option<*mut BasicPausible>, // only one is allowed
    remotes: ~[(uint, ~Callback)],
    next_remote: uint,
    messages: Exclusive<~[Message]>
}

enum Message { RunRemote(uint), RemoveRemote(uint) }

struct Time {
    sec: u64,
    nsec: u64,
}

impl Ord for Time {
    fn lt(&self, other: &Time) -> bool {
        self.sec < other.sec || self.nsec < other.nsec
    }
}

impl BasicLoop {
    fn new() -> BasicLoop {
        BasicLoop {
            work: ~[],
            idle: None,
            next_remote: 0,
            remotes: ~[],
            messages: Exclusive::new(~[]),
        }
    }

    /// Process everything in the work queue (continually)
    fn work(&mut self) {
        while self.work.len() > 0 {
            for work in util::replace(&mut self.work, ~[]).move_iter() {
                work();
            }
        }
    }

    fn remote_work(&mut self) {
        let messages = unsafe {
            do self.messages.with |messages| {
                if messages.len() > 0 {
                    Some(util::replace(messages, ~[]))
                } else {
                    None
                }
            }
        };
        let messages = match messages {
            Some(m) => m, None => return
        };
        for message in messages.iter() {
            self.message(*message);
        }
    }

    fn message(&mut self, message: Message) {
        match message {
            RunRemote(i) => {
                match self.remotes.mut_iter().find(|& &(id, _)| id == i) {
                    Some(&(_, ref mut f)) => f.call(),
                    None => unreachable!()
                }
            }
            RemoveRemote(i) => {
                match self.remotes.iter().position(|&(id, _)| id == i) {
                    Some(i) => { self.remotes.remove(i); }
                    None => unreachable!()
                }
            }
        }
    }

    /// Run the idle callback if one is registered
    fn idle(&mut self) {
        unsafe {
            match self.idle {
                Some(idle) => {
                    if (*idle).active {
                        (*idle).work.call();
                    }
                }
                None => {}
            }
        }
    }

    fn has_idle(&self) -> bool {
        unsafe { self.idle.is_some() && (**self.idle.get_ref()).active }
    }
}

impl EventLoop for BasicLoop {
    fn run(&mut self) {
        // Not exactly efficient, but it gets the job done.
        while self.remotes.len() > 0 || self.work.len() > 0 || self.has_idle() {

            self.work();
            self.remote_work();

            if self.has_idle() {
                self.idle();
                continue
            }

            unsafe {
                // We block here if we have no messages to process and we may
                // receive a message at a later date
                do self.messages.hold_and_wait |messages| {
                    self.remotes.len() > 0 &&
                        messages.len() == 0 &&
                        self.work.len() == 0
                }
            }
        }
    }

    fn callback(&mut self, f: proc()) {
        self.work.push(f);
    }

    // XXX: Seems like a really weird requirement to have an event loop provide.
    fn pausible_idle_callback(&mut self, cb: ~Callback) -> ~PausibleIdleCallback {
        let callback = ~BasicPausible::new(self, cb);
        rtassert!(self.idle.is_none());
        unsafe {
            let cb_ptr: &*mut BasicPausible = cast::transmute(&callback);
            self.idle = Some(*cb_ptr);
        }
        return callback as ~PausibleIdleCallback;
    }

    fn remote_callback(&mut self, f: ~Callback) -> ~RemoteCallback {
        let id = self.next_remote;
        self.next_remote += 1;
        self.remotes.push((id, f));
        ~BasicRemote::new(self.messages.clone(), id) as ~RemoteCallback
    }

    /// This has no bindings for local I/O
    fn io<'a>(&'a mut self, _: &fn(&'a mut IoFactory)) {}
}

struct BasicRemote {
    queue: Exclusive<~[Message]>,
    id: uint,
}

impl BasicRemote {
    fn new(queue: Exclusive<~[Message]>, id: uint) -> BasicRemote {
        BasicRemote { queue: queue, id: id }
    }
}

impl RemoteCallback for BasicRemote {
    fn fire(&mut self) {
        unsafe {
            do self.queue.hold_and_signal |queue| {
                queue.push(RunRemote(self.id));
            }
        }
    }
}

impl Drop for BasicRemote {
    fn drop(&mut self) {
        unsafe {
            do self.queue.hold_and_signal |queue| {
                queue.push(RemoveRemote(self.id));
            }
        }
    }
}

struct BasicPausible {
    eloop: *mut BasicLoop,
    work: ~Callback,
    active: bool,
}

impl BasicPausible {
    fn new(eloop: &mut BasicLoop, cb: ~Callback) -> BasicPausible {
        BasicPausible {
            active: false,
            work: cb,
            eloop: eloop,
        }
    }
}

impl PausibleIdleCallback for BasicPausible {
    fn pause(&mut self) {
        self.active = false;
    }
    fn resume(&mut self) {
        self.active = true;
    }
}

impl Drop for BasicPausible {
    fn drop(&mut self) {
        unsafe {
            (*self.eloop).idle = None;
        }
    }
}

fn time() -> Time {
    #[fixed_stack_segment]; #[inline(never)];
    extern {
        fn get_time(sec: &mut i64, nsec: &mut i32);
    }
    let mut sec = 0;
    let mut nsec = 0;
    unsafe { get_time(&mut sec, &mut nsec) }

    Time { sec: sec as u64, nsec: nsec as u64 }
}
