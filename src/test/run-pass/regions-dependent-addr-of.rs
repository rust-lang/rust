// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test lifetimes are linked properly when we create dependent region pointers.
// Issue #3148.


struct A {
    value: B
}

struct B {
    v1: int,
    v2: [int, ..3],
    v3: Vec<int> ,
    v4: C,
    v5: Box<C>,
    v6: Option<C>
}

struct C {
    f: int
}

fn get_v1<'v>(a: &'v A) -> &'v int {
    // Region inferencer must deduce that &v < L2 < L1
    let foo = &a.value; // L1
    &foo.v1             // L2
}

fn get_v2<'v>(a: &'v A, i: uint) -> &'v int {
    let foo = &a.value;
    &foo.v2[i]
}

fn get_v3<'v>(a: &'v A, i: uint) -> &'v int {
    let foo = &a.value;
    foo.v3.get(i)
}

fn get_v4<'v>(a: &'v A, _i: uint) -> &'v int {
    let foo = &a.value;
    &foo.v4.f
}

fn get_v5<'v>(a: &'v A, _i: uint) -> &'v int {
    let foo = &a.value;
    &foo.v5.f
}

fn get_v6_a<'v>(a: &'v A, _i: uint) -> &'v int {
    match a.value.v6 {
        Some(ref v) => &v.f,
        None => fail!()
    }
}

fn get_v6_b<'v>(a: &'v A, _i: uint) -> &'v int {
    match *a {
        A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => fail!()
    }
}

fn get_v6_c<'v>(a: &'v A, _i: uint) -> &'v int {
    match a {
        &A { value: B { v6: Some(ref v), .. } } => &v.f,
        _ => fail!()
    }
}

fn get_v5_ref<'v>(a: &'v A, _i: uint) -> &'v int {
    match &a.value {
        &B {v5: box C {f: ref v}, ..} => v
    }
}

pub fn main() {
    let a = A {value: B {v1: 22,
                         v2: [23, 24, 25],
                         v3: vec!(26, 27, 28),
                         v4: C { f: 29 },
                         v5: box C { f: 30 },
                         v6: Some(C { f: 31 })}};

    let p = get_v1(&a);
    assert_eq!(*p, a.value.v1);

    let p = get_v2(&a, 1);
    assert_eq!(*p, a.value.v2[1]);

    let p = get_v3(&a, 1);
    assert_eq!(*p, *a.value.v3.get(1));

    let p = get_v4(&a, 1);
    assert_eq!(*p, a.value.v4.f);

    let p = get_v5(&a, 1);
    assert_eq!(*p, a.value.v5.f);

    let p = get_v6_a(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_b(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v6_c(&a, 1);
    assert_eq!(*p, a.value.v6.unwrap().f);

    let p = get_v5_ref(&a, 1);
    assert_eq!(*p, a.value.v5.f);
}
