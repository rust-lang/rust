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
use std::mem;
use std::rt::bookkeeping;
use std::rt::local::Local;
use std::rt::mutex::NativeMutex;
use std::rt::rtio;
use std::rt::stack;
use std::rt::task::{Task, BlockedTask, TaskOpts};
use std::rt::thread::Thread;
use std::rt;

use io;
use task;
use std::task::{TaskBuilder, Spawner};

/// Creates a new Task which is ready to execute as a 1:1 task.
pub fn new(stack_bounds: (uint, uint)) -> Box<Task> {
    let mut task = box Task::new();
    let mut ops = ops();
    ops.stack_bounds = stack_bounds;
    task.put_runtime(ops);
    return task;
}

fn ops() -> Box<Ops> {
    box Ops {
        lock: unsafe { NativeMutex::new() },
        awoken: false,
        io: io::IoFactory::new(),
        // these *should* get overwritten
        stack_bounds: (0, 0),
    }
}

/// Spawns a function with the default configuration
#[deprecated = "use the native method of NativeTaskBuilder instead"]
pub fn spawn(f: proc():Send) {
    spawn_opts(TaskOpts { name: None, stack_size: None, on_exit: None }, f)
}

/// Spawns a new task given the configuration options and a procedure to run
/// inside the task.
#[deprecated = "use the native method of NativeTaskBuilder instead"]
pub fn spawn_opts(opts: TaskOpts, f: proc():Send) {
    let TaskOpts { name, stack_size, on_exit } = opts;

    let mut task = box Task::new();
    task.name = name;
    task.death.on_exit = on_exit;

    let stack = stack_size.unwrap_or(rt::min_stack());
    let task = task;
    let ops = ops();

    // Note that this increment must happen *before* the spawn in order to
    // guarantee that if this task exits it will always end up waiting for the
    // spawned task to exit.
    let token = bookkeeping::increment();

    // Spawning a new OS thread guarantees that __morestack will never get
    // triggered, but we must manually set up the actual stack bounds once this
    // function starts executing. This raises the lower limit by a bit because
    // by the time that this function is executing we've already consumed at
    // least a little bit of stack (we don't know the exact byte address at
    // which our stack started).
    Thread::spawn_stack(stack, proc() {
        let something_around_the_top_of_the_stack = 1;
        let addr = &something_around_the_top_of_the_stack as *const int;
        let my_stack = addr as uint;
        unsafe {
            stack::record_os_managed_stack_bounds(my_stack - stack + 1024, my_stack);
        }
        let mut ops = ops;
        ops.stack_bounds = (my_stack - stack + 1024, my_stack);

        let mut f = Some(f);
        let mut task = task;
        task.put_runtime(ops);
        drop(task.run(|| { f.take().unwrap()() }).destroy());
        drop(token);
    })
}

/// A spawner for native tasks
pub struct NativeSpawner;

impl Spawner for NativeSpawner {
    fn spawn(self, opts: TaskOpts, f: proc():Send) {
        spawn_opts(opts, f)
    }
}

/// An extension trait adding a `native` configuration method to `TaskBuilder`.
pub trait NativeTaskBuilder {
    fn native(self) -> TaskBuilder<NativeSpawner>;
}

impl<S: Spawner> NativeTaskBuilder for TaskBuilder<S> {
    fn native(self) -> TaskBuilder<NativeSpawner> {
        self.spawner(NativeSpawner)
    }
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
    fn yield_now(self: Box<Ops>, mut cur_task: Box<Task>) {
        // put the task back in TLS and then invoke the OS thread yield
        cur_task.put_runtime(self);
        Local::put(cur_task);
        Thread::yield_now();
    }

    fn maybe_yield(self: Box<Ops>, mut cur_task: Box<Task>) {
        // just put the task back in TLS, on OS threads we never need to
        // opportunistically yield b/c the OS will do that for us (preemption)
        cur_task.put_runtime(self);
        Local::put(cur_task);
    }

    fn wrap(self: Box<Ops>) -> Box<Any+'static> {
        self as Box<Any+'static>
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
    // to a `Box<Task>` object. Naturally, this looks like it violates
    // ownership semantics in that there may be two `Box<Task>` objects.
    //
    // The fun part is that the wakeup half of this implementation knows to
    // "forget" the task on the other end. This means that the awakening half of
    // things silently relinquishes ownership back to this thread, but not in a
    // way that the compiler can understand. The task's memory is always valid
    // for both tasks because these operations are all done inside of a mutex.
    //
    // You'll also find that if blocking fails (the `f` function hands the
    // BlockedTask back to us), we will `mem::forget` the handles. The
    // reasoning for this is the same logic as above in that the task silently
    // transfers ownership via the `uint`, not through normal compiler
    // semantics.
    //
    // On a mildly unrelated note, it should also be pointed out that OS
    // condition variables are susceptible to spurious wakeups, which we need to
    // be ready for. In order to accommodate for this fact, we have an extra
    // `awoken` field which indicates whether we were actually woken up via some
    // invocation of `reawaken`. This flag is only ever accessed inside the
    // lock, so there's no need to make it atomic.
    fn deschedule(mut self: Box<Ops>,
                  times: uint,
                  mut cur_task: Box<Task>,
                  f: |BlockedTask| -> Result<(), BlockedTask>) {
        let me = &mut *self as *mut Ops;
        cur_task.put_runtime(self);

        unsafe {
            let cur_task_dupe = &mut *cur_task as *mut Task;
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
                    Err(task) => { mem::forget(task.wake()); }
                }
            } else {
                let iter = task.make_selectable(times);
                let guard = (*me).lock.lock();
                (*me).awoken = false;

                // Apply the given closure to all of the "selectable tasks",
                // bailing on the first one that produces an error. Note that
                // care must be taken such that when an error is occurred, we
                // may not own the task, so we may still have to wait for the
                // task to become available. In other words, if task.wake()
                // returns `None`, then someone else has ownership and we must
                // wait for their signal.
                match iter.map(f).filter_map(|a| a.err()).next() {
                    None => {}
                    Some(task) => {
                        match task.wake() {
                            Some(task) => {
                                mem::forget(task);
                                (*me).awoken = true;
                            }
                            None => {}
                        }
                    }
                }
                while !(*me).awoken {
                    guard.wait();
                }
            }
            // re-acquire ownership of the task
            cur_task = mem::transmute(cur_task_dupe);
        }

        // put the task back in TLS, and everything is as it once was.
        Local::put(cur_task);
    }

    // See the comments on `deschedule` for why the task is forgotten here, and
    // why it's valid to do so.
    fn reawaken(mut self: Box<Ops>, mut to_wake: Box<Task>) {
        unsafe {
            let me = &mut *self as *mut Ops;
            to_wake.put_runtime(self);
            mem::forget(to_wake);
            let guard = (*me).lock.lock();
            (*me).awoken = true;
            guard.signal();
        }
    }

    fn spawn_sibling(self: Box<Ops>,
                     mut cur_task: Box<Task>,
                     opts: TaskOpts,
                     f: proc():Send) {
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
    use std::rt::task::{Task, TaskOpts};
    use std::task;
    use std::task::TaskBuilder;
    use super::{spawn, spawn_opts, Ops, NativeTaskBuilder};

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
        assert_eq!(rx.recv_opt(), Err(()));
    }

    #[test]
    fn smoke_opts() {
        let mut opts = TaskOpts::new();
        opts.name = Some("test".into_maybe_owned());
        opts.stack_size = Some(20 * 4096);
        let (tx, rx) = channel();
        opts.on_exit = Some(proc(r) tx.send(r));
        spawn_opts(opts, proc() {});
        assert!(rx.recv().is_ok());
    }

    #[test]
    fn smoke_opts_fail() {
        let mut opts = TaskOpts::new();
        let (tx, rx) = channel();
        opts.on_exit = Some(proc(r) tx.send(r));
        spawn_opts(opts, proc() { fail!() });
        assert!(rx.recv().is_err());
    }

    #[test]
    fn yield_test() {
        let (tx, rx) = channel();
        spawn(proc() {
            for _ in range(0u, 10) { task::deschedule(); }
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
                let mut task: Box<Task> = Local::take();
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

    #[test]
    fn test_native_builder() {
        let res = TaskBuilder::new().native().try(proc() {
            "Success!".to_string()
        });
        assert_eq!(res.ok().unwrap(), "Success!".to_string());
    }
}
