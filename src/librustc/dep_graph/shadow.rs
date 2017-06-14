// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The "Shadow Graph" is maintained on the main thread and which
//! tracks each message relating to the dep-graph and applies some
//! sanity checks as they go by. If an error results, it means you get
//! a nice stack-trace telling you precisely what caused the error.
//!
//! NOTE: This is a debugging facility which can potentially have non-trivial
//! runtime impact. Therefore, it is largely compiled out if
//! debug-assertions are not enabled.
//!
//! The basic sanity check, enabled if you have debug assertions
//! enabled, is that there is always a task (or ignore) on the stack
//! when you do read/write, and that the tasks are pushed/popped
//! according to a proper stack discipline.
//!
//! Optionally, if you specify RUST_FORBID_DEP_GRAPH_EDGE, you can
//! specify an edge filter to be applied to each edge as it is
//! created.  See `./README.md` for details.

use std::cell::RefCell;

use super::DepNode;
use super::thread::DepMessage;

pub struct ShadowGraph {
    // if you push None onto the stack, that corresponds to an Ignore
    stack: RefCell<Vec<Option<DepNode>>>,
}

const ENABLED: bool = cfg!(debug_assertions);

impl ShadowGraph {
    pub fn new() -> Self {
        ShadowGraph {
            stack: RefCell::new(vec![]),
        }
    }

    #[inline]
    pub fn enabled(&self) -> bool {
        ENABLED
    }

    pub fn enqueue(&self, message: &DepMessage) {
        if ENABLED {
            if self.stack.try_borrow().is_err() {
                // When we apply edge filters, that invokes the Debug trait on
                // DefIds, which in turn reads from various bits of state and
                // creates reads! Ignore those recursive reads.
                return;
            }

            let mut stack = self.stack.borrow_mut();
            match *message {
                // It is ok to READ shared state outside of a
                // task. That can't do any harm (at least, the only
                // way it can do harm is by leaking that data into a
                // query or task, which would be a problem
                // anyway). What would be bad is WRITING to that
                // state.
                DepMessage::Read(_) => { }
                DepMessage::PushTask(ref n) => stack.push(Some(n.clone())),
                DepMessage::PushIgnore => stack.push(None),
                DepMessage::PopTask(ref n) => {
                    match stack.pop() {
                        Some(Some(m)) => {
                            if *n != m {
                                bug!("stack mismatch: found {:?} expected {:?}", m, n)
                            }
                        }
                        Some(None) => bug!("stack mismatch: found Ignore expected {:?}", n),
                        None => bug!("stack mismatch: found empty stack, expected {:?}", n),
                    }
                }
                DepMessage::PopIgnore => {
                    match stack.pop() {
                        Some(Some(m)) => bug!("stack mismatch: found {:?} expected ignore", m),
                        Some(None) => (),
                        None => bug!("stack mismatch: found empty stack, expected ignore"),
                    }
                }
                DepMessage::Query => (),
            }
        }
    }
}
