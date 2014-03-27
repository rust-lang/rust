// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tasks implemented on top of OS threads
//!
//! This module contains the implementation of the 1:1 threading module required
//! by rust tasks. This implements the necessary API traits laid out by std::rt
//! in order to spawn new tasks and deschedule the current task.

use std::any::Any;
use std::cast;
use std::rt::bookkeeping;
use std::rt::env;
use std::rt::local::Local;
use std::rt::rtio;
use std::rt::stack;
use std::rt::task::{Task, BlockedTask, SendMessage};
use std::rt::thread::Thread;
use std::rt;
use std::task::TaskOpts;
use std::unstable::mutex::NativeMutex;

use io;
use task;

/// Creates a new Task which is ready to execute as a 1:1 task.
pub fn new(stack_bounds: (uint, uint)) -> ~Task {
    let mut task = ~Task::new();
    let mut ops = ops();
    ops.stack_bounds = stack_bounds;
    task.put_runtime(ops);
    return task;
}

fn ops() -> ~Ops {
    ~Ops {
        lock: unsafe { NativeMutex::new() },
        awoken: false,
        io: io::IoFactory::new(),
        // these *should* get overwritten
        stack_bounds: (0, 0),
    }
}

/// Spawns a function with the default configuration
pub fn spawn(f: proc:Send()) {
    spawn_opts(TaskOpts::new(), f)
}

/// Spawns a new task given the configuration options and a procedure to run
/// inside the task.
pub fn spawn_opts(opts: TaskOpts, f: proc:Send()) {
    let TaskOpts {
        notify_chan, name, stack_size,
        stderr, stdout,
    } = opts;

    let mut task = ~Task::new();
    task.name = name;
    task.stderr = stderr;
    task.stdout = stdout;
    match notify_chan {
        Some(chan) => { task.death.on_exit = Some(SendMessage(chan)); }
        None => {}
    }

    let stack = stack_size.unwrap_or(env::min_stack());
    let task = task;
    let ops = ops();

    // Note that this increment must happen *before* the spawn in order to
    // guarantee that if this task exits it will always end up waiting for the
    // spawned task to exit.
    bookkeeping::increment();

    // Spawning a new OS thread guarantees that __morestack will never get
    // triggered, but we must manually set up the actual stack bounds once this
    // function starts executing. This raises the lower limit by a bit because
    // by the time that this function is executing we've already consumed at
    // least a little bit of stack (we don't know the exact byte address at
    // which our stack started).
    Thread::spawn_stack(stack, proc() {
        let something_around_the_top_of_the_stack = 1;
        let addr = &something_around_the_top_of_the_stack as *int;
        let my_stack = addr as uint;
        unsafe {
            stack::record_stack_bounds(my_stack - stack + 1024, my_stack);
        }
        let mut ops = ops;
        ops.stack_bounds = (my_stack - stack + 1024, my_stack);

        let mut f = Some(f);
        let mut task = task;
        task.put_runtime(ops);
        let t = task.run(|| { f.take_unwrap()() });
        drop(t);
        bookkeeping::decrement();
    })
}

// This structure is the glue between channels and the 1:1 scheduling mode. This
// structure is allocated once per task.
struct Ops {
    lock: NativeMutex,       // native synchronization
    awoken: bool,      // used to prevent spurious wakeups
    io: io::IoFactory, // local I/O factory

    // This field holds the known bounds of the stack in (lo, hi) form. Not all
    // native tasks necessarily know their precise bounds, hence this is
    // optional.
    stack_bounds: (uint, uint),
}

impl rt::Runtime for Ops {
    fn yield_now(~self, mut cur_task: ~Task) {
        // put the task back in TLS and then invoke the OS thread yield
        cur_task.put_runtime(self);
        Local::put(cur_task);
        Thread::yield_now();
    }

    fn maybe_yield(~self, mut cur_task: ~Task) {
        // just put the task back in TLS, on OS threads we never need to
        // opportunistically yield b/c the OS will do that for us (preemption)
        cur_task.put_runtime(self);
        Local::put(cur_task);
    }

    fn wrap(~self) -> ~Any {
        self as ~Any
    }

    fn stack_bounds(&self) -> (uint, uint) { self.stack_bounds }

    fn can_block(&self) -> bool { true }

    // This function gets a little interesting. There are a few safety and
    // ownership violations going on here, but this is all done in the name of
    // shared state. Additionally, all of the violations are protected with a
    // mutex, so in theory there are no races.
    //
    // The first thing we need to do is to get a pointer to the task's internal
    // mutex. This address will not be changing (because the task is allocated
    // on the heap). We must have this handle separately because the task will
    // have its ownership transferred to the given closure. We're guaranteed,
    // however, that this memory will remain valid because *this* is the current
    // task's execution thread.
    //
    // The next weird part is where ownership of the task actually goes. We
    // relinquish it to the `f` blocking function, but upon returning this
    // function needs to replace the task back in TLS. There is no communication
    // from the wakeup thread back to this thread about the task pointer, and
    // there's really no need to. In order to get around this, we cast the task
    // to a `uint` which is then used at the end of this function to cast back
    // to a `~Task` object. Naturally, this looks like it violates ownership
    // semantics in that there may be two `~Task` objects.
    //
    // The fun part is that the wakeup half of this implementation knows to
    // "forget" the task on the other end. This means that the awakening half of
    // things silently relinquishes ownership back to this thread, but not in a
    // way that the compiler can understand. The task's memory is always valid
    // for both tasks because these operations are all done inside of a mutex.
    //
    // You'll also find that if blocking fails (the `f` function hands the
    // BlockedTask back to us), we will `cast::forget` the handles. The
    // reasoning for this is the same logic as above in that the task silently
    // transfers ownership via the `uint`, not through normal compiler
    // semantics.
    //
    // On a mildly unrelated note, it should also be pointed out that OS
    // condition variables are susceptible to spurious wakeups, which we need to
    // be ready for. In order to accomodate for this fact, we have an extra
    // `awoken` field which indicates whether we were actually woken up via some
    // invocation of `reawaken`. This flag is only ever accessed inside the
    // lock, so there's no need to make it atomic.
    fn deschedule(mut ~self, times: uint, mut cur_task: ~Task,
                  f: |BlockedTask| -> Result<(), BlockedTask>) {
        let me = &mut *self as *mut Ops;
        cur_task.put_runtime(self);

        unsafe {
            let cur_task_dupe = &*cur_task as *Task;
            let task = BlockedTask::block(cur_task);

            if times == 1 {
                let guard = (*me).lock.lock();
                (*me).awoken = false;
                match f(task) {
                    Ok(()) => {
                        while !(*me).awoken {
                            guard.wait();
                        }
                    }
                    Err(task) => { cast::forget(task.wake()); }
                }
            } else {
                let mut iter = task.make_selectable(times);
                let guard = (*me).lock.lock();
                (*me).awoken = false;
                let success = iter.all(|task| {
                    match f(task) {
                        Ok(()) => true,
                        Err(task) => {
                            cast::forget(task.wake());
                            false
                        }
                    }
                });
                while success && !(*me).awoken {
                    guard.wait();
                }
            }
            // re-acquire ownership of the task
            cur_task = cast::transmute(cur_task_dupe);
        }

        // put the task back in TLS, and everything is as it once was.
        Local::put(cur_task);
    }

    // See the comments on `deschedule` for why the task is forgotten here, and
    // why it's valid to do so.
    fn reawaken(mut ~self, mut to_wake: ~Task) {
        unsafe {
            let me = &mut *self as *mut Ops;
            to_wake.put_runtime(self);
            cast::forget(to_wake);
            let guard = (*me).lock.lock();
            (*me).awoken = true;
            guard.signal();
        }
    }

    fn spawn_sibling(~self, mut cur_task: ~Task, opts: TaskOpts, f: proc:Send()) {
        cur_task.put_runtime(self);
        Local::put(cur_task);

        task::spawn_opts(opts, f);
    }

    fn local_io<'a>(&'a mut self) -> Option<rtio::LocalIo<'a>> {
        Some(rtio::LocalIo::new(&mut self.io as &mut rtio::IoFactory))
    }
}

#[cfg(test)]
mod tests {
    use std::rt::local::Local;
    use std::rt::task::Task;
    use std::task;
    use std::task::TaskOpts;
    use super::{spawn, spawn_opts, Ops};

    #[test]
    fn smoke() {
        let (tx, rx) = channel();
        spawn(proc() {
            tx.send(());
        });
        rx.recv();
    }

    #[test]
    fn smoke_fail() {
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let _tx = tx;
            fail!()
        });
        assert_eq!(rx.recv_opt(), None);
    }

    #[test]
    fn smoke_opts() {
        let mut opts = TaskOpts::new();
        opts.name = Some("test".into_maybe_owned());
        opts.stack_size = Some(20 * 4096);
        let (tx, rx) = channel();
        opts.notify_chan = Some(tx);
        spawn_opts(opts, proc() {});
        assert!(rx.recv().is_ok());
    }

    #[test]
    fn smoke_opts_fail() {
        let mut opts = TaskOpts::new();
        let (tx, rx) = channel();
        opts.notify_chan = Some(tx);
        spawn_opts(opts, proc() { fail!() });
        assert!(rx.recv().is_err());
    }

    #[test]
    fn yield_test() {
        let (tx, rx) = channel();
        spawn(proc() {
            for _ in range(0, 10) { task::deschedule(); }
            tx.send(());
        });
        rx.recv();
    }

    #[test]
    fn spawn_children() {
        let (tx1, rx) = channel();
        spawn(proc() {
            let (tx2, rx) = channel();
            spawn(proc() {
                let (tx3, rx) = channel();
                spawn(proc() {
                    tx3.send(());
                });
                rx.recv();
                tx2.send(());
            });
            rx.recv();
            tx1.send(());
        });
        rx.recv();
    }

    #[test]
    fn spawn_inherits() {
        let (tx, rx) = channel();
        spawn(proc() {
            spawn(proc() {
                let mut task: ~Task = Local::take();
                match task.maybe_take_runtime::<Ops>() {
                    Some(ops) => {
                        task.put_runtime(ops);
                    }
                    None => fail!(),
                }
                Local::put(task);
                tx.send(());
            });
        });
        rx.recv();
    }
}
