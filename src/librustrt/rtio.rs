// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The EventLoop and internal synchronous I/O interface.

use core::prelude::*;
use alloc::boxed::Box;

pub trait EventLoop {
    fn run(&mut self);
    fn callback(&mut self, arg: proc(): Send);
    fn pausable_idle_callback(&mut self, Box<Callback + Send>)
                              -> Box<PausableIdleCallback + Send>;
    fn remote_callback(&mut self, Box<Callback + Send>)
                       -> Box<RemoteCallback + Send>;

    // last vestige of IoFactory
    fn has_active_io(&self) -> bool;
}

pub trait Callback {
    fn call(&mut self);
}

pub trait RemoteCallback {
    /// Trigger the remote callback. Note that the number of times the
    /// callback is run is not guaranteed. All that is guaranteed is
    /// that, after calling 'fire', the callback will be called at
    /// least once, but multiple callbacks may be coalesced and
    /// callbacks may be called more often requested. Destruction also
    /// triggers the callback.
    fn fire(&mut self);
}

pub trait PausableIdleCallback {
    fn pause(&mut self);
    fn resume(&mut self);
}
