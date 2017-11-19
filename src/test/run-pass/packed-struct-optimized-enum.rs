// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(packed)]
#[derive(Copy, Clone)]
struct Packed<T>(T);

fn main() {
    let one = (Some(Packed((&(), 0))), true);
    let two = [one, one];
    let stride = (&two[1] as *const _ as usize) - (&two[0] as *const _ as usize);

    // This can fail if rustc and LLVM disagree on the size of a type.
    // In this case, `Option<Packed<(&(), u32)>>` was erronously not
    // marked as packed despite needing alignment `1` and containing
    // its `&()` discriminant, which has alignment larger than `1`.
    assert_eq!(stride, std::mem::size_of_val(&one));
}
