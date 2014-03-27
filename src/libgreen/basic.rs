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
//!
//! This implementation is also used as the fallback implementation of an event
//! loop if no other one is provided (and M:N scheduling is desired).

use std::cast;
use std::mem::replace;
use std::rt::rtio::{EventLoop, IoFactory, RemoteCallback, PausableIdleCallback,
                    Callback};
use std::unstable::sync::Exclusive;

/// This is the only exported function from this module.
pub fn event_loop() -> ~EventLoop:Send {
    ~BasicLoop::new() as ~EventLoop:Send
}

struct BasicLoop {
    work: ~[proc:Send()],             // pending work
    idle: Option<*mut BasicPausable>, // only one is allowed
    remotes: ~[(uint, ~Callback:Send)],
    next_remote: uint,
    messages: Exclusive<~[Message]>,
}

enum Message { RunRemote(uint), RemoveRemote(uint) }

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
            for work in replace(&mut self.work, ~[]).move_iter() {
                work();
            }
        }
    }

    fn remote_work(&mut self) {
        let messages = unsafe {
            self.messages.with(|messages| {
                if messages.len() > 0 {
                    Some(replace(messages, ~[]))
                } else {
                    None
                }
            })
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
                    Some(i) => { self.remotes.remove(i).unwrap(); }
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
                self.messages.hold_and_wait(|messages| {
                    self.remotes.len() > 0 &&
                        messages.len() == 0 &&
                        self.work.len() == 0
                })
            }
        }
    }

    fn callback(&mut self, f: proc:Send()) {
        self.work.push(f);
    }

    // FIXME: Seems like a really weird requirement to have an event loop provide.
    fn pausable_idle_callback(&mut self, cb: ~Callback:Send)
        -> ~PausableIdleCallback:Send
    {
        let callback = ~BasicPausable::new(self, cb);
        rtassert!(self.idle.is_none());
        unsafe {
            let cb_ptr: &*mut BasicPausable = cast::transmute(&callback);
            self.idle = Some(*cb_ptr);
        }
        callback as ~PausableIdleCallback:Send
    }

    fn remote_callback(&mut self, f: ~Callback:Send) -> ~RemoteCallback:Send {
        let id = self.next_remote;
        self.next_remote += 1;
        self.remotes.push((id, f));
        ~BasicRemote::new(self.messages.clone(), id) as ~RemoteCallback:Send
    }

    fn io<'a>(&'a mut self) -> Option<&'a mut IoFactory> { None }

    fn has_active_io(&self) -> bool { false }
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
            self.queue.hold_and_signal(|queue| {
                queue.push(RunRemote(self.id));
            })
        }
    }
}

impl Drop for BasicRemote {
    fn drop(&mut self) {
        unsafe {
            self.queue.hold_and_signal(|queue| {
                queue.push(RemoveRemote(self.id));
            })
        }
    }
}

struct BasicPausable {
    eloop: *mut BasicLoop,
    work: ~Callback:Send,
    active: bool,
}

impl BasicPausable {
    fn new(eloop: &mut BasicLoop, cb: ~Callback:Send) -> BasicPausable {
        BasicPausable {
            active: false,
            work: cb,
            eloop: eloop,
        }
    }
}

impl PausableIdleCallback for BasicPausable {
    fn pause(&mut self) {
        self.active = false;
    }
    fn resume(&mut self) {
        self.active = true;
    }
}

impl Drop for BasicPausable {
    fn drop(&mut self) {
        unsafe {
            (*self.eloop).idle = None;
        }
    }
}

#[cfg(test)]
mod test {
    use std::task::TaskOpts;

    use basic;
    use PoolConfig;
    use SchedPool;

    fn pool() -> SchedPool {
        SchedPool::new(PoolConfig {
            threads: 1,
            event_loop_factory: basic::event_loop,
        })
    }

    fn run(f: proc()) {
        let mut pool = pool();
        pool.spawn(TaskOpts::new(), f);
        pool.shutdown();
    }

    #[test]
    fn smoke() {
        run(proc() {});
    }

    #[test]
    fn some_channels() {
        run(proc() {
            let (tx, rx) = channel();
            spawn(proc() {
                tx.send(());
            });
            rx.recv();
        });
    }

    #[test]
    fn multi_thread() {
        let mut pool = SchedPool::new(PoolConfig {
            threads: 2,
            event_loop_factory: basic::event_loop,
        });

        for _ in range(0, 20) {
            pool.spawn(TaskOpts::new(), proc() {
                let (tx, rx) = channel();
                spawn(proc() {
                    tx.send(());
                });
                rx.recv();
            });
        }

        pool.shutdown();
    }
}
