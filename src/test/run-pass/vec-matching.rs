// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(advanced_slice_patterns)]
#![feature(slice_patterns)]

fn a() {
    let x = [1];
    match x {
        [a] => {
            assert_eq!(a, 1);
        }
    }
}

fn b() {
    let x = [1, 2, 3];
    match x {
        [a, b, c..] => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            let expected: &[_] = &[3];
            assert_eq!(c, expected);
        }
    }
    match x {
        [a.., b, c] => {
            let expected: &[_] = &[1];
            assert_eq!(a, expected);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
    match x {
        [a, b.., c] => {
            assert_eq!(a, 1);
            let expected: &[_] = &[2];
            assert_eq!(b, expected);
            assert_eq!(c, 3);
        }
    }
    match x {
        [a, b, c] => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
}


fn b_slice() {
    let x : &[_] = &[1, 2, 3];
    match x {
        &[a, b, ref c..] => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            let expected: &[_] = &[3];
            assert_eq!(c, expected);
        }
        _ => unreachable!()
    }
    match x {
        &[ref a.., b, c] => {
            let expected: &[_] = &[1];
            assert_eq!(a, expected);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
        _ => unreachable!()
    }
    match x {
        &[a, ref b.., c] => {
            assert_eq!(a, 1);
            let expected: &[_] = &[2];
            assert_eq!(b, expected);
            assert_eq!(c, 3);
        }
        _ => unreachable!()
    }
    match x {
        &[a, b, c] => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
        _ => unreachable!()
    }
}

fn c() {
    let x = [1];
    match x {
        [2, ..] => panic!(),
        [..] => ()
    }
}

fn d() {
    let x = [1, 2, 3];
    let branch = match x {
        [1, 1, ..] => 0,
        [1, 2, 3, ..] => 1,
        [1, 2, ..] => 2,
        _ => 3
    };
    assert_eq!(branch, 1);
}

fn e() {
    let x: &[isize] = &[1, 2, 3];
    let a = match *x {
        [1, 2] => 0,
        [..] => 1,
    };

    assert_eq!(a, 1);

    let b = match *x {
        [2, ..] => 0,
        [1, 2, ..] => 1,
        [_] => 2,
        [..] => 3
    };

    assert_eq!(b, 1);


    let c = match *x {
        [_, _, _, _, ..] => 0,
        [1, 2, ..] => 1,
        [_] => 2,
        [..] => 3
    };

    assert_eq!(c, 1);
}

fn f() {
    let x = &[1, 2, 3, 4, 5];
    let [a, [b, [c, ..].., d].., e] = *x;
    assert_eq!((a, b, c, d, e), (1, 2, 3, 4, 5));

    let x: &[isize] = x;
    let (a, b, c, d, e) = match *x {
        [a, [b, [c, ..].., d].., e] => (a, b, c, d, e),
        _ => unimplemented!()
    };

    assert_eq!((a, b, c, d, e), (1, 2, 3, 4, 5));
}

pub fn main() {
    a();
    b();
    b_slice();
    c();
    d();
    e();
    f();
}
