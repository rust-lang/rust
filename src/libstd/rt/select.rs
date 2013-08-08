// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Module for private, abstraction-leaking select traits. Wrapped in std::select.

use rt::kill::BlockedTask;
use rt::sched::Scheduler;
use option::Option;

pub trait SelectInner {
    // Returns true if data was available.
    fn optimistic_check(&mut self) -> bool;
    // Returns true if data was available. If so, shall also wake() the task.
    fn block_on(&mut self, &mut Scheduler, BlockedTask) -> bool;
    // Returns true if data was available.
    fn unblock_from(&mut self) -> bool;
}

pub trait SelectPortInner<T> {
    fn recv_ready(self) -> Option<T>;
}

