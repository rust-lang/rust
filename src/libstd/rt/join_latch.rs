// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The JoinLatch is a concurrent type that establishes the task
//! tree and propagates failure.
//!
//! Each task gets a JoinLatch that is derived from the JoinLatch
//! of its parent task. Every latch must be released by either calling
//! the non-blocking `release` method or the task-blocking `wait` method.
//! Releasing a latch does not complete until all of its child latches
//! complete.
//!
//! Latches carry a `success` flag that is set to `false` during task
//! failure and is propagated both from children to parents and parents
//! to children. The status af this flag may be queried for the purposes
//! of linked failure.
//!
//! In addition to failure propagation the task tree serves to keep the
//! default task schedulers alive. The runtime only sends the shutdown
//! message to schedulers once the root task exits.
//!
//! Under this scheme tasks that terminate before their children become
//! 'zombies' since they may not exit until their children do. Zombie
//! tasks are 'tombstoned' as `Tombstone(~JoinLatch)` and the tasks
//! themselves allowed to terminate.
//!
//! XXX: Propagate flag from parents to children.
//! XXX: Tombstoning actually doesn't work.
//! XXX: This could probably be done in a way that doesn't leak tombstones
//!      longer than the life of the child tasks.

use comm::{GenericPort, Peekable, GenericSmartChan};
use clone::Clone;
use container::Container;
use option::{Option, Some, None};
use ops::Drop;
use rt::comm::{SharedChan, Port, stream};
use rt::local::Local;
use rt::sched::Scheduler;
use unstable::atomics::{AtomicUint, SeqCst};
use util;
use vec::OwnedVector;

// FIXME #7026: Would prefer this to be an enum
pub struct JoinLatch {
    priv parent: Option<ParentLink>,
    priv child: Option<ChildLink>,
    closed: bool,
}

// Shared between parents and all their children.
struct SharedState {
    /// Reference count, held by a parent and all children.
    count: AtomicUint,
    success: bool
}

struct ParentLink {
    shared: *mut SharedState,
    // For communicating with the parent.
    chan: SharedChan<Message>
}

struct ChildLink {
    shared: ~SharedState,
    // For receiving from children.
    port: Port<Message>,
    chan: SharedChan<Message>,
    // Prevents dropping the child SharedState reference counts multiple times.
    dropped_child: bool
}

// Messages from child latches to parent.
enum Message {
    Tombstone(~JoinLatch),
    ChildrenTerminated
}

impl JoinLatch {
    pub fn new_root() -> ~JoinLatch {
        let this = ~JoinLatch {
            parent: None,
            child: None,
            closed: false
        };
        rtdebug!("new root latch %x", this.id());
        return this;
    }

    fn id(&self) -> uint {
        unsafe { ::cast::transmute(&*self) }
    }

    pub fn new_child(&mut self) -> ~JoinLatch {
        rtassert!(!self.closed);

        if self.child.is_none() {
            // This is the first time spawning a child
            let shared = ~SharedState {
                count: AtomicUint::new(1),
                success: true
            };
            let (port, chan) = stream();
            let chan = SharedChan::new(chan);
            let child = ChildLink {
                shared: shared,
                port: port,
                chan: chan,
                dropped_child: false
            };
            self.child = Some(child);
        }

        let child_link: &mut ChildLink = self.child.get_mut_ref();
        let shared_state: *mut SharedState = &mut *child_link.shared;

        child_link.shared.count.fetch_add(1, SeqCst);

        let child = ~JoinLatch {
            parent: Some(ParentLink {
                shared: shared_state,
                chan: child_link.chan.clone()
            }),
            child: None,
            closed: false
        };
        rtdebug!("NEW child latch %x", child.id());
        return child;
    }

    pub fn release(~self, local_success: bool) {
        // XXX: This should not block, but there's a bug in the below
        // code that I can't figure out.
        self.wait(local_success);
    }

    // XXX: Should not require ~self
    fn release_broken(~self, local_success: bool) {
        rtassert!(!self.closed);

        rtdebug!("releasing %x", self.id());

        let id = self.id();
        let _ = id; // XXX: `id` is only used in debug statements so appears unused
        let mut this = self;
        let mut child_success = true;
        let mut children_done = false;

        if this.child.is_some() {
            rtdebug!("releasing children");
            let child_link: &mut ChildLink = this.child.get_mut_ref();
            let shared: &mut SharedState = &mut *child_link.shared;

            if !child_link.dropped_child {
                let last_count = shared.count.fetch_sub(1, SeqCst);
                rtdebug!("child count before sub %u %x", last_count, id);
                if last_count == 1 {
                    assert!(child_link.chan.try_send(ChildrenTerminated));
                }
                child_link.dropped_child = true;
            }

            // Wait for messages from children
            let mut tombstones = ~[];
            loop {
                if child_link.port.peek() {
                    match child_link.port.recv() {
                        Tombstone(t) => {
                            tombstones.push(t);
                        },
                        ChildrenTerminated => {
                            children_done = true;
                            break;
                        }
                    }
                } else {
                    break
                }
            }

            rtdebug!("releasing %u tombstones %x", tombstones.len(), id);

            // Try to release the tombstones. Those that still have
            // outstanding will be re-enqueued.  When this task's
            // parents release their latch we'll end up back here
            // trying them again.
            while !tombstones.is_empty() {
                tombstones.pop().release(true);
            }

            if children_done {
                let count = shared.count.load(SeqCst);
                assert!(count == 0);
                // self_count is the acquire-read barrier
                child_success = shared.success;
            }
        } else {
            children_done = true;
        }

        let total_success = local_success && child_success;

        rtassert!(this.parent.is_some());

        unsafe {
            {
                let parent_link: &mut ParentLink = this.parent.get_mut_ref();
                let shared: *mut SharedState = parent_link.shared;

                if !total_success {
                    // parent_count is the write-wait barrier
                    (*shared).success = false;
                }
            }

            if children_done {
                rtdebug!("children done");
                do Local::borrow::<Scheduler, ()> |sched| {
                    sched.metrics.release_tombstone += 1;
                }
                {
                    rtdebug!("RELEASING parent %x", id);
                    let parent_link: &mut ParentLink = this.parent.get_mut_ref();
                    let shared: *mut SharedState = parent_link.shared;
                    let last_count = (*shared).count.fetch_sub(1, SeqCst);
                    rtdebug!("count before parent sub %u %x", last_count, id);
                    if last_count == 1 {
                        assert!(parent_link.chan.try_send(ChildrenTerminated));
                    }
                }
                this.closed = true;
                util::ignore(this);
            } else {
                rtdebug!("children not done");
                rtdebug!("TOMBSTONING %x", id);
                do Local::borrow::<Scheduler, ()> |sched| {
                    sched.metrics.release_no_tombstone += 1;
                }
                let chan = {
                    let parent_link: &mut ParentLink = this.parent.get_mut_ref();
                    parent_link.chan.clone()
                };
                assert!(chan.try_send(Tombstone(this)));
            }
        }
    }

    // XXX: Should not require ~self
    pub fn wait(~self, local_success: bool) -> bool {
        rtassert!(!self.closed);

        rtdebug!("WAITING %x", self.id());

        let mut this = self;
        let mut child_success = true;

        if this.child.is_some() {
            rtdebug!("waiting for children");
            let child_link: &mut ChildLink = this.child.get_mut_ref();
            let shared: &mut SharedState = &mut *child_link.shared;

            if !child_link.dropped_child {
                let last_count = shared.count.fetch_sub(1, SeqCst);
                rtdebug!("child count before sub %u", last_count);
                if last_count == 1 {
                    assert!(child_link.chan.try_send(ChildrenTerminated));
                }
                child_link.dropped_child = true;
            }

            // Wait for messages from children
            loop {
                match child_link.port.recv() {
                    Tombstone(t) => {
                        t.wait(true);
                    }
                    ChildrenTerminated => break
                }
            }

            let count = shared.count.load(SeqCst);
            if count != 0 { ::io::println(fmt!("%u", count)); }
            assert!(count == 0);
            // self_count is the acquire-read barrier
            child_success = shared.success;
        }

        let total_success = local_success && child_success;

        if this.parent.is_some() {
            rtdebug!("releasing parent");
            unsafe {
                let parent_link: &mut ParentLink = this.parent.get_mut_ref();
                let shared: *mut SharedState = parent_link.shared;

                if !total_success {
                    // parent_count is the write-wait barrier
                    (*shared).success = false;
                }

                let last_count = (*shared).count.fetch_sub(1, SeqCst);
                rtdebug!("count before parent sub %u", last_count);
                if last_count == 1 {
                    assert!(parent_link.chan.try_send(ChildrenTerminated));
                }
            }
        }

        this.closed = true;
        util::ignore(this);

        return total_success;
    }
}

impl Drop for JoinLatch {
    fn drop(&self) {
        rtdebug!("DESTROYING %x", self.id());
        rtassert!(self.closed);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cell::Cell;
    use container::Container;
    use iter::Times;
    use rt::test::*;
    use rand;
    use rand::RngUtil;
    use vec::{CopyableVector, ImmutableVector};

    #[test]
    fn success_immediately() {
        do run_in_newsched_task {
            let mut latch = JoinLatch::new_root();

            let child_latch = latch.new_child();
            let child_latch = Cell::new(child_latch);
            do spawntask_immediately {
                let child_latch = child_latch.take();
                assert!(child_latch.wait(true));
            }

            assert!(latch.wait(true));
        }
    }

    #[test]
    fn success_later() {
        do run_in_newsched_task {
            let mut latch = JoinLatch::new_root();

            let child_latch = latch.new_child();
            let child_latch = Cell::new(child_latch);
            do spawntask_later {
                let child_latch = child_latch.take();
                assert!(child_latch.wait(true));
            }

            assert!(latch.wait(true));
        }
    }

    #[test]
    fn mt_success() {
        do run_in_mt_newsched_task {
            let mut latch = JoinLatch::new_root();

            for 10.times {
                let child_latch = latch.new_child();
                let child_latch = Cell::new(child_latch);
                do spawntask_random {
                    let child_latch = child_latch.take();
                    assert!(child_latch.wait(true));
                }
            }

            assert!(latch.wait(true));
        }
    }

    #[test]
    fn mt_failure() {
        do run_in_mt_newsched_task {
            let mut latch = JoinLatch::new_root();

            let spawn = |status| {
                let child_latch = latch.new_child();
                let child_latch = Cell::new(child_latch);
                do spawntask_random {
                    let child_latch = child_latch.take();
                    child_latch.wait(status);
                }
            };

            for 10.times { spawn(true) }
            spawn(false);
            for 10.times { spawn(true) }

            assert!(!latch.wait(true));
        }
    }

    #[test]
    fn mt_multi_level_success() {
        do run_in_mt_newsched_task {
            let mut latch = JoinLatch::new_root();

            fn child(latch: &mut JoinLatch, i: int) {
                let child_latch = latch.new_child();
                let child_latch = Cell::new(child_latch);
                do spawntask_random {
                    let mut child_latch = child_latch.take();
                    if i != 0 {
                        child(&mut *child_latch, i - 1);
                        child_latch.wait(true);
                    } else {
                        child_latch.wait(true);
                    }
                }
            }

            child(&mut *latch, 10);

            assert!(latch.wait(true));
        }
    }

    #[test]
    fn mt_multi_level_failure() {
        do run_in_mt_newsched_task {
            let mut latch = JoinLatch::new_root();

            fn child(latch: &mut JoinLatch, i: int) {
                let child_latch = latch.new_child();
                let child_latch = Cell::new(child_latch);
                do spawntask_random {
                    let mut child_latch = child_latch.take();
                    if i != 0 {
                        child(&mut *child_latch, i - 1);
                        child_latch.wait(false);
                    } else {
                        child_latch.wait(true);
                    }
                }
            }

            child(&mut *latch, 10);

            assert!(!latch.wait(true));
        }
    }

    #[test]
    fn release_child() {
        do run_in_newsched_task {
            let mut latch = JoinLatch::new_root();
            let child_latch = latch.new_child();
            let child_latch = Cell::new(child_latch);

            do spawntask_immediately {
                let latch = child_latch.take();
                latch.release(false);
            }

            assert!(!latch.wait(true));
        }
    }

    #[test]
    fn release_child_tombstone() {
        do run_in_newsched_task {
            let mut latch = JoinLatch::new_root();
            let child_latch = latch.new_child();
            let child_latch = Cell::new(child_latch);

            do spawntask_immediately {
                let mut latch = child_latch.take();
                let child_latch = latch.new_child();
                let child_latch = Cell::new(child_latch);
                do spawntask_later {
                    let latch = child_latch.take();
                    latch.release(false);
                }
                latch.release(true);
            }

            assert!(!latch.wait(true));
        }
    }

    #[test]
    fn release_child_no_tombstone() {
        do run_in_newsched_task {
            let mut latch = JoinLatch::new_root();
            let child_latch = latch.new_child();
            let child_latch = Cell::new(child_latch);

            do spawntask_later {
                let mut latch = child_latch.take();
                let child_latch = latch.new_child();
                let child_latch = Cell::new(child_latch);
                do spawntask_immediately {
                    let latch = child_latch.take();
                    latch.release(false);
                }
                latch.release(true);
            }

            assert!(!latch.wait(true));
        }
    }

    #[test]
    fn release_child_tombstone_stress() {
        fn rand_orders() -> ~[bool] {
            let mut v = ~[false,.. 5];
            v[0] = true;
            let mut rng = rand::rng();
            return rng.shuffle(v);
        }

        fn split_orders(orders: &[bool]) -> (~[bool], ~[bool]) {
            if orders.is_empty() {
                return (~[], ~[]);
            } else if orders.len() <= 2 {
                return (orders.to_owned(), ~[]);
            }
            let mut rng = rand::rng();
            let n = rng.gen_uint_range(1, orders.len());
            let first = orders.slice(0, n).to_owned();
            let last = orders.slice(n, orders.len()).to_owned();
            assert!(first.len() + last.len() == orders.len());
            return (first, last);
        }

        for stress_factor().times {
            do run_in_newsched_task {
                fn doit(latch: &mut JoinLatch, orders: ~[bool], depth: uint) {
                    let (my_orders, remaining_orders) = split_orders(orders);
                    rtdebug!("(my_orders, remaining): %?", (&my_orders, &remaining_orders));
                    rtdebug!("depth: %u", depth);
                    let mut remaining_orders = remaining_orders;
                    let mut num = 0;
                    for my_orders.iter().advance |&order| {
                        let child_latch = latch.new_child();
                        let child_latch = Cell::new(child_latch);
                        let (child_orders, remaining) = split_orders(remaining_orders);
                        rtdebug!("(child_orders, remaining): %?", (&child_orders, &remaining));
                        remaining_orders = remaining;
                        let child_orders = Cell::new(child_orders);
                        let child_num = num;
                        let _ = child_num; // XXX unused except in rtdebug!
                        do spawntask_random {
                            rtdebug!("depth %u num %u", depth, child_num);
                            let mut child_latch = child_latch.take();
                            let child_orders = child_orders.take();
                            doit(&mut *child_latch, child_orders, depth + 1);
                            child_latch.release(order);
                        }

                        num += 1;
                    }
                }

                let mut latch = JoinLatch::new_root();
                let orders = rand_orders();
                rtdebug!("orders: %?", orders);

                doit(&mut *latch, orders, 0);

                assert!(!latch.wait(true));
            }
        }
    }

    #[deriving(Clone)]
    struct Order {
        immediate: bool,
        succeed: bool,
        orders: ~[Order]
    }

    #[test]
    fn whateverman() {
        fn next(latch: &mut JoinLatch, orders: ~[Order]) {
            for orders.iter().advance |order| {
                let suborders = order.orders.clone();
                let child_latch = Cell::new(latch.new_child());
                let succeed = order.succeed;
                if order.immediate {
                    do spawntask_immediately {
                        let mut child_latch = child_latch.take();
                        next(&mut *child_latch, suborders.clone());
                        rtdebug!("immediate releasing");
                        child_latch.release(succeed);
                    }
                } else {
                    do spawntask_later {
                        let mut child_latch = child_latch.take();
                        next(&mut *child_latch, suborders.clone());
                        rtdebug!("later releasing");
                        child_latch.release(succeed);
                    }
                }
            }
        }

        do run_in_newsched_task {
            let mut latch = JoinLatch::new_root();
            let orders = ~[ Order { // 0 0
                immediate: true,
                succeed: true,
                orders: ~[ Order { // 1 0
                    immediate: true,
                    succeed: false,
                    orders: ~[ Order { // 2 0
                        immediate: false,
                        succeed: false,
                        orders: ~[ Order { // 3 0
                            immediate: true,
                            succeed: false,
                            orders: ~[]
                        }, Order { // 3 1
                            immediate: false,
                            succeed: false,
                            orders: ~[]
                        }]
                    }]
                }]
            }];

            next(&mut *latch, orders);
            assert!(!latch.wait(true));
        }
    }
}

