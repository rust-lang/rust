// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// A trait for unlockable locks.
pub trait Unlock {
    /**
     * Temporarily unlocks a lock.
     *
     * # Safety notes
     * In order to guarantee safety for users, and implementors of
     * Unlock the following restrictions are made.
     *
     * - The passed in closure is owned so that it cannot refer to
     *   borrowed values such as the lock itself.
     *
     * - The passed in closure is once to force that it is used at
     *   most once.
     *
     * - The U parameter forces unlock to fail, or use the passed in
     *   closure at least once.
     *
     * - The self pointer is mutable so as to invalidate references to
     *   internal state of the lock.
     */
    fn unlock<U>(&mut self, ~once fn() -> U) -> U;
}