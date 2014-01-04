// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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

use std::cast;
use std::rt::env;
use std::rt::local::Local;
use std::rt::rtio;
use std::rt::task::{Task, BlockedTask};
use std::rt::thread::Thread;
use std::rt;
use std::task::TaskOpts;
use std::unstable::mutex::Mutex;
use std::unstable::stack;

use io;
use task;
use bookeeping;

/// Creates a new Task which is ready to execute as a 1:1 task.
pub fn new() -> ~Task {
    let mut task = ~Task::new();
    task.put_runtime(ops() as ~rt::Runtime);
    return task;
}

fn ops() -> ~Ops {
    ~Ops {
        lock: unsafe { Mutex::new() },
        awoken: false,
        io: io::IoFactory::new(),
        stack_bounds: None,
    }
}

/// Spawns a function with the default configuration
pub fn spawn(f: proc()) {
    spawn_opts(TaskOpts::new(), f)
}

/// Spawns a new task given the configuration options and a procedure to run
/// inside the task.
pub fn spawn_opts(opts: TaskOpts, f: proc()) {
    let TaskOpts {
        watched: _watched,
        notify_chan, name, stack_size
    } = opts;

    let mut task = ~Task::new();
    task.name = name;
    match notify_chan {
        Some(chan) => {
            let on_exit = proc(task_result) { chan.send(task_result) };
            task.death.on_exit = Some(on_exit);
        }
        None => {}
    }

    let stack = stack_size.unwrap_or(env::min_stack());
    let task = task;
    let ops = ops();

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
        ops.stack_bounds = Some((my_stack - stack + 1024, my_stack));

        bookeeping::increment();
        let mut f = Some(f);
        let mut task = task;
        task.put_runtime(ops as ~rt::Runtime);
        task.run(|| { f.take_unwrap()() });
        bookeeping::decrement();
    })
}

// This structure is the glue between channels and the 1:1 scheduling mode. This
// structure is allocated once per task.
struct Ops {
    lock: Mutex,       // native synchronization
    awoken: bool,      // used to prevent spurious wakeups
    io: io::IoFactory, // local I/O factory

    // This field holds the known bounds of the stack in (lo, hi) form. Not all
    // native tasks necessarily know their precise bounds, hence this is
    // optional.
    stack_bounds: Option<(uint, uint)>,
}

impl rt::Runtime for Ops {
    fn yield_now(~self, mut cur_task: ~Task) {
        // put the task back in TLS and then invoke the OS thread yield
        cur_task.put_runtime(self as ~rt::Runtime);
        Local::put(cur_task);
        Thread::yield_now();
    }

    fn maybe_yield(~self, mut cur_task: ~Task) {
        // just put the task back in TLS, on OS threads we never need to
        // opportunistically yield b/c the OS will do that for us (preemption)
        cur_task.put_runtime(self as ~rt::Runtime);
        Local::put(cur_task);
    }

    fn wrap(~self) -> ~Any {
        self as ~Any
    }

    fn stack_bounds(&self) -> Option<(uint, uint)> { self.stack_bounds }

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
        cur_task.put_runtime(self as ~rt::Runtime);

        unsafe {
            let cur_task_dupe = *cast::transmute::<&~Task, &uint>(&cur_task);
            let task = BlockedTask::block(cur_task);

            if times == 1 {
                (*me).lock.lock();
                (*me).awoken = false;
                match f(task) {
                    Ok(()) => {
                        while !(*me).awoken {
                            (*me).lock.wait();
                        }
                    }
                    Err(task) => { cast::forget(task.wake()); }
                }
                (*me).lock.unlock();
            } else {
                let mut iter = task.make_selectable(times);
                (*me).lock.lock();
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
                    (*me).lock.wait();
                }
                (*me).lock.unlock();
            }
            // re-acquire ownership of the task
            cur_task = cast::transmute::<uint, ~Task>(cur_task_dupe);
        }

        // put the task back in TLS, and everything is as it once was.
        Local::put(cur_task);
    }

    // See the comments on `deschedule` for why the task is forgotten here, and
    // why it's valid to do so.
    fn reawaken(mut ~self, mut to_wake: ~Task, _can_resched: bool) {
        unsafe {
            let me = &mut *self as *mut Ops;
            to_wake.put_runtime(self as ~rt::Runtime);
            cast::forget(to_wake);
            (*me).lock.lock();
            (*me).awoken = true;
            (*me).lock.signal();
            (*me).lock.unlock();
        }
    }

    fn spawn_sibling(~self, mut cur_task: ~Task, opts: TaskOpts, f: proc()) {
        cur_task.put_runtime(self as ~rt::Runtime);
        Local::put(cur_task);

        task::spawn_opts(opts, f);
    }

    fn local_io<'a>(&'a mut self) -> Option<rtio::LocalIo<'a>> {
        Some(rtio::LocalIo::new(&mut self.io as &mut rtio::IoFactory))
    }
}

impl Drop for Ops {
    fn drop(&mut self) {
        unsafe { self.lock.destroy() }
    }
}

#[cfg(test)]
mod tests {
    use std::rt::Runtime;
    use std::rt::local::Local;
    use std::rt::task::Task;
    use std::task;
    use std::task::TaskOpts;
    use super::{spawn, spawn_opts, Ops};

    #[test]
    fn smoke() {
        let (p, c) = Chan::new();
        do spawn {
            c.send(());
        }
        p.recv();
    }

    #[test]
    fn smoke_fail() {
        let (p, c) = Chan::<()>::new();
        do spawn {
            let _c = c;
            fail!()
        }
        assert_eq!(p.recv_opt(), None);
    }

    #[test]
    fn smoke_opts() {
        let mut opts = TaskOpts::new();
        opts.name = Some(SendStrStatic("test"));
        opts.stack_size = Some(20 * 4096);
        let (p, c) = Chan::new();
        opts.notify_chan = Some(c);
        spawn_opts(opts, proc() {});
        assert!(p.recv().is_ok());
    }

    #[test]
    fn smoke_opts_fail() {
        let mut opts = TaskOpts::new();
        let (p, c) = Chan::new();
        opts.notify_chan = Some(c);
        spawn_opts(opts, proc() { fail!() });
        assert!(p.recv().is_err());
    }

    #[test]
    fn yield_test() {
        let (p, c) = Chan::new();
        do spawn {
            10.times(task::deschedule);
            c.send(());
        }
        p.recv();
    }

    #[test]
    fn spawn_children() {
        let (p, c) = Chan::new();
        do spawn {
            let (p, c2) = Chan::new();
            do spawn {
                let (p, c3) = Chan::new();
                do spawn {
                    c3.send(());
                }
                p.recv();
                c2.send(());
            }
            p.recv();
            c.send(());
        }
        p.recv();
    }

    #[test]
    fn spawn_inherits() {
        let (p, c) = Chan::new();
        do spawn {
            let c = c;
            do spawn {
                let mut task: ~Task = Local::take();
                match task.maybe_take_runtime::<Ops>() {
                    Some(ops) => {
                        task.put_runtime(ops as ~Runtime);
                    }
                    None => fail!(),
                }
                Local::put(task);
                c.send(());
            }
        }
        p.recv();
    }
}
