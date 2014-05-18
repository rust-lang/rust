// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let a = ary![42i, ..1];
    assert_eq!(a.as_slice(), [42i].as_slice());
    assert_eq!(a.as_slice(), [42i, ..1].as_slice());

    let b = ary![42i, ..10];
    assert_eq!(b.as_slice(), [42i, ..10].as_slice());

    let c = ary![None::<Vec<u8>>, ..10];
    assert_eq!(c.len(), 10);
    assert_eq!(c.as_slice(), Vec::from_elem(10, None::<Vec<u8>>).as_slice());

    #[deriving(Eq)]
    // non-Copy, non-Clone
    struct Foo {
        x: uint,
        nocopy: ::std::kinds::marker::NoCopy
    }

    impl Foo {
        fn new(x: uint) -> Foo {
            Foo {
                x: x,
                nocopy: ::std::kinds::marker::NoCopy
            }
        }
    }

    let d = ary![Foo::new(42), ..10];
    assert_eq!(d.len(), 10);
    assert!(d.as_slice() == Vec::from_fn(10, |_| Foo::new(42)).as_slice());

    let mut it = range(0, 4);
    let e = ary![it.next(), ..8];
    assert_eq!(e.as_slice(), &[Some(0), Some(1), Some(2), Some(3), None, None, None, None]);

    let f = ary![format!("{}", 42i), ..4];
    assert_eq!(f.len(), 4);
    for s in f.iter() {
        assert_eq!(s.as_slice(), "42");
    }

    let g = ary![vec![1u,2,3], ..4];
    assert_eq!(g.len(), 4);
    for v in g.iter() {
        assert_eq!(v.as_slice(), &[1u,2,3]);
    }
}
