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
struct Packed<T: Copy>(T);

impl<T: Copy> Copy for Packed<T> {}
impl<T: Copy> Clone for Packed<T> {
    fn clone(&self) -> Self { *self }
}

fn sanity_check_size<T: Copy>(one: T) {
    let two = [one, one];
    let stride = (&two[1] as *const _ as usize) - (&two[0] as *const _ as usize);
    let (size, align) = (std::mem::size_of::<T>(), std::mem::align_of::<T>());
    assert_eq!(stride, size);
    assert_eq!(size % align, 0);
}

fn main() {
    // This can fail if rustc and LLVM disagree on the size of a type.
    // In this case, `Option<Packed<(&(), u32)>>` was erronously not
    // marked as packed despite needing alignment `1` and containing
    // its `&()` discriminant, which has alignment larger than `1`.
    sanity_check_size((Some(Packed((&(), 0))), true));

    // In #46769, `Option<(Packed<&()>, bool)>` was found to have
    // pointer alignment, without actually being aligned in size.
    // E.g. on 64-bit platforms, it had alignment `8` but size `9`.
    type PackedRefAndBool<'a> = (Packed<&'a ()>, bool);
    sanity_check_size::<Option<PackedRefAndBool>>(Some((Packed(&()), true)));

    // Make sure we don't pay for the enum optimization in size,
    // e.g. we shouldn't need extra padding after the packed data.
    assert_eq!(std::mem::align_of::<Option<PackedRefAndBool>>(), 1);
    assert_eq!(std::mem::size_of::<Option<PackedRefAndBool>>(),
               std::mem::size_of::<PackedRefAndBool>());
}
