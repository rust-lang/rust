// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Exposes the `ManuallyDrop` lang item for controlling the exactly what gets dropped within a
//! structure.

/// A wrapper type that stores data inline without running its destructor when dropped.
#[lang="manually_drop"]
#[experimental]
pub struct ManuallyDrop<T> {
    /// Wrapped value
    ///
    /// This field should not be accessed directly, it is made public for static
    /// initializers.
    #[unstable]
    pub value: T,
}

impl<T> ManuallyDrop<T> {
    /// Construct a new instance of `ManuallyDrop` which will wrap the specified
    /// value.
    ///
    /// All access to the inner value through methods is `unsafe`, and it is
    /// highly discouraged to access the fields directly.
    ///
    /// This function is unsafe for the same reason as `forget`, namely that it
    /// prevents the value's destructors from running.
    #[experimental]
    pub unsafe fn new(value: T) -> ManuallyDrop<T> {
        ManuallyDrop { value: value }
    }

    /// Gets a mutable pointer to the wrapped value.
    ///
    /// This function is unsafe as the pointer returned is an unsafe pointer and
    /// no guarantees are made about the aliasing of the pointers being handed
    /// out in this or other tasks.
    #[experimental]
    pub unsafe fn get(&self) -> *const T {
        &self.value as *const T
    }

    /// Unwraps the value
    ///
    /// This function is unsafe because there is no guarantee that this or other
    /// tasks are currently inspecting the inner value. Additionally, the destructor
    /// for the inner value is suddenly rescheduled to run.
    #[experimental]
    pub unsafe fn into_inner(self) -> T { self.value }
}
