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
use std::rt::stack;
use std::rt::task::{Task, BlockedTask, TaskOpts};
use std::rt::thread::Thread;
use std::rt;

#[cfg(test)]
mod tests {
    use std::rt::local::Local;
    use std::rt::task::{Task, TaskOpts};
    use std::task;
    use std::task::{TaskBuilder, Spawner};

    use super::{Ops, NativeTaskBuilder, NativeSpawner};

    #[test]
    fn smoke() {
        let (tx, rx) = channel();
        spawn(proc() {
            tx.send(());
        });
        rx.recv();
    }

    #[test]
    fn smoke_panic() {
        let (tx, rx) = channel::<()>();
        spawn(proc() {
            let _tx = tx;
            panic!()
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
        NativeSpawner.spawn(opts, proc() {});
        assert!(rx.recv().is_ok());
    }

    #[test]
    fn smoke_opts_panic() {
        let mut opts = TaskOpts::new();
        let (tx, rx) = channel();
        opts.on_exit = Some(proc(r) tx.send(r));
        NativeSpawner.spawn(opts, proc() { panic!() });
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
        TaskBuilder::new().spawner(NativeSpawner).spawn(proc() {
            spawn(proc() {
                let mut task: Box<Task> = Local::take();
                match task.maybe_take_runtime::<Ops>() {
                    Some(ops) => {
                        task.put_runtime(ops);
                    }
                    None => panic!(),
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
