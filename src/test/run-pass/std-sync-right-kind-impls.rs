// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(static_mutex, static_rwlock, static_condvar)]
#![feature(semaphore)]

use std::sync;

fn assert_both<T: Sync + Send>() {}

fn main() {
    assert_both::<sync::Mutex<()>>();
    assert_both::<sync::Condvar>();
    assert_both::<sync::RwLock<()>>();
    assert_both::<sync::Barrier>();
    assert_both::<sync::Arc<()>>();
    assert_both::<sync::Weak<()>>();
    assert_both::<sync::Once>();
}
