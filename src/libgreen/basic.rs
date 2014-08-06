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

use alloc::arc::Arc;
use std::sync::atomic;
use std::mem;
use std::rt::rtio::{EventLoop, IoFactory, RemoteCallback};
use std::rt::rtio::{PausableIdleCallback, Callback};
use std::rt::exclusive::Exclusive;

/// This is the only exported function from this module.
pub fn event_loop() -> Box<EventLoop + Send> {
    box BasicLoop::new() as Box<EventLoop + Send>
}

struct BasicLoop {
    work: Vec<proc(): Send>,             // pending work
    remotes: Vec<(uint, Box<Callback + Send>)>,
    next_remote: uint,
    messages: Arc<Exclusive<Vec<Message>>>,
    idle: Option<Box<Callback + Send>>,
    idle_active: Option<Arc<atomic::AtomicBool>>,
}

enum Message { RunRemote(uint), RemoveRemote(uint) }

impl BasicLoop {
    fn new() -> BasicLoop {
        BasicLoop {
            work: vec![],
            idle: None,
            idle_active: None,
            next_remote: 0,
            remotes: vec![],
            messages: Arc::new(Exclusive::new(Vec::new())),
        }
    }

    /// Process everything in the work queue (continually)
    fn work(&mut self) {
        while self.work.len() > 0 {
            for work in mem::replace(&mut self.work, vec![]).move_iter() {
                work();
            }
        }
    }

    fn remote_work(&mut self) {
        let messages = unsafe {
            mem::replace(&mut *self.messages.lock(), Vec::new())
        };
        for message in messages.move_iter() {
            self.message(message);
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
        match self.idle {
            Some(ref mut idle) => {
                if self.idle_active.get_ref().load(atomic::SeqCst) {
                    idle.call();
                }
            }
            None => {}
        }
    }

    fn has_idle(&self) -> bool {
        self.idle.is_some() && self.idle_active.get_ref().load(atomic::SeqCst)
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
                let mut messages = self.messages.lock();
                // We block here if we have no messages to process and we may
                // receive a message at a later date
                if self.remotes.len() > 0 && messages.len() == 0 &&
                   self.work.len() == 0 {
                    messages.wait()
                }
            }
        }
    }

    fn callback(&mut self, f: proc():Send) {
        self.work.push(f);
    }

    // FIXME: Seems like a really weird requirement to have an event loop provide.
    fn pausable_idle_callback(&mut self, cb: Box<Callback + Send>)
                              -> Box<PausableIdleCallback + Send> {
        rtassert!(self.idle.is_none());
        self.idle = Some(cb);
        let a = Arc::new(atomic::AtomicBool::new(true));
        self.idle_active = Some(a.clone());
        box BasicPausable { active: a } as Box<PausableIdleCallback + Send>
    }

    fn remote_callback(&mut self, f: Box<Callback + Send>)
                       -> Box<RemoteCallback + Send> {
        let id = self.next_remote;
        self.next_remote += 1;
        self.remotes.push((id, f));
        box BasicRemote::new(self.messages.clone(), id) as
            Box<RemoteCallback + Send>
    }

    fn io<'a>(&'a mut self) -> Option<&'a mut IoFactory> { None }

    fn has_active_io(&self) -> bool { false }
}

struct BasicRemote {
    queue: Arc<Exclusive<Vec<Message>>>,
    id: uint,
}

impl BasicRemote {
    fn new(queue: Arc<Exclusive<Vec<Message>>>, id: uint) -> BasicRemote {
        BasicRemote { queue: queue, id: id }
    }
}

impl RemoteCallback for BasicRemote {
    fn fire(&mut self) {
        let mut queue = unsafe { self.queue.lock() };
        queue.push(RunRemote(self.id));
        queue.signal();
    }
}

impl Drop for BasicRemote {
    fn drop(&mut self) {
        let mut queue = unsafe { self.queue.lock() };
        queue.push(RemoveRemote(self.id));
        queue.signal();
    }
}

struct BasicPausable {
    active: Arc<atomic::AtomicBool>,
}

impl PausableIdleCallback for BasicPausable {
    fn pause(&mut self) {
        self.active.store(false, atomic::SeqCst);
    }
    fn resume(&mut self) {
        self.active.store(true, atomic::SeqCst);
    }
}

impl Drop for BasicPausable {
    fn drop(&mut self) {
        self.active.store(false, atomic::SeqCst);
    }
}

#[cfg(test)]
mod test {
    use std::rt::task::TaskOpts;

    use basic;
    use PoolConfig;
    use SchedPool;

    fn pool() -> SchedPool {
        SchedPool::new(PoolConfig {
            threads: 1,
            event_loop_factory: basic::event_loop,
        })
    }

    fn run(f: proc():Send) {
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

        for _ in range(0u, 20) {
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
