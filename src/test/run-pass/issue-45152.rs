// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unsize, coerce_unsized)]

#[repr(packed)]
struct UnalignedPtr<'a, T: ?Sized>
    where T: 'a,
{
    data: &'a T,
}

fn main() {

    impl<'a, T, U> std::ops::CoerceUnsized<UnalignedPtr<'a, U>> for UnalignedPtr<'a, T>
        where
        T: std::marker::Unsize<U> + ?Sized,
        U: ?Sized,
    { }

    let arr = [1, 2, 3];
    let arr_unaligned: UnalignedPtr<[i32; 3]> = UnalignedPtr { data: &arr };
    let arr_unaligned: UnalignedPtr<[i32]> = arr_unaligned;
}
