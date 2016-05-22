// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use core::array::FixedSizeArray;

#[test]
fn fixed_size_array() {
    let mut array = [0; 64];
    let mut zero_sized = [(); 64];
    let mut empty_array = [0; 0];
    let mut empty_zero_sized = [(); 0];

    assert_eq!(FixedSizeArray::as_slice(&array).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_slice(&empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_slice(&empty_zero_sized).len(), 0);

    assert_eq!(FixedSizeArray::as_mut_slice(&mut array).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut zero_sized).len(), 64);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_array).len(), 0);
    assert_eq!(FixedSizeArray::as_mut_slice(&mut empty_zero_sized).len(), 0);
}

#[test]
fn fixed_size_array_conversions() {
    let mut empty: [u32; 0] = [];
    let () = empty.into();
    let &() = empty.as_ref();
    let &mut () = empty.as_mut();

    let mut foo: [u8; 3] = *b"foo";
    let (a, b, c) = foo.into();
    assert!((a, b, c) == (b'f', b'o', b'o'));

    let &(a, b, c) = foo.as_ref();
    assert!((a, b, c) == (b'f', b'o', b'o'));

    {
        let &mut (ref mut a, b, c) = foo.as_mut();
        assert!((*a, b, c) == (b'f', b'o', b'o'));
        *a = b'F';
    }
    assert!(&foo == b"Foo");
}
