// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Sweep {
    fn sweep(&self) -> usize;
}

trait SliceWrapper<T> {
    fn slice(&self) -> &[T];
}

trait SliceWrapperMut<T> {
    fn slice_mut(&mut self) -> &mut [T];
}

fn foo<B: SliceWrapper<u32> + SliceWrapperMut<u32> + Sweep>(mut buckets: B) {
    let key = 0u32;
    buckets.slice_mut()[(key as usize).wrapping_add(22).wrapping_rem(buckets.sweep())] = 22;
    //~^ ERROR cannot borrow `buckets` as immutable
}

fn main() { }
