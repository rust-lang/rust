// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Eq, TotalEq, Ord, TotalOrd)]
struct S<T> {
    x: T,
    y: T
}

#[deriving(Eq, TotalEq, Ord, TotalOrd)]
struct TS<T>(T,T);

#[deriving(Eq, TotalEq, Ord, TotalOrd)]
enum E<T> {
    E0,
    E1(T),
    E2(T,T)
}

#[deriving(Eq, TotalEq, Ord, TotalOrd)]
enum ES<T> {
    ES1 { x: T },
    ES2 { x: T, y: T }
}


pub fn main() {
    let s1 = S {x: 1, y: 1}, s2 = S {x: 1, y: 2};
    let ts1 = TS(1, 1), ts2 = TS(1,2);
    let e0 = E0, e11 = E1(1), e12 = E1(2), e21 = E2(1,1), e22 = E2(1, 2);
    let es11 = ES1 {x: 1}, es12 = ES1 {x: 2}, es21 = ES2 {x: 1, y: 1}, es22 = ES2 {x: 1, y: 2};

    test([s1, s2]);
    test([ts1, ts2]);
    test([e0, e11, e12, e21, e22]);
    test([es11, es12, es21, es22]);
}

fn test<T: Eq+TotalEq+Ord+TotalOrd>(ts: &[T]) {
    // compare each element against all other elements. The list
    // should be in sorted order, so that if i < j, then ts[i] <
    // ts[j], etc.
    for ts.eachi |i, t1| {
        for ts.eachi |j, t2| {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j, le = i <= j;
            let gt = i > j, ge = i >= j;

            // Eq
            assert_eq!(*t1 == *t2, eq);

            // TotalEq
            assert_eq!(t1.equals(t2), eq);

            // Ord
            assert_eq!(*t1 < *t2, lt);
            assert_eq!(*t1 > *t2, gt);

            assert_eq!(*t1 <= *t2, le);
            assert_eq!(*t1 >= *t2, ge);

            // TotalOrd
            assert_eq!(t1.cmp(t2), ord);
        }
    }
}