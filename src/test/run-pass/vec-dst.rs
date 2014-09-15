// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate debug;

fn reflect() {
    // Tests for reflective printing.
    // Also tests drop glue.
    let x = [1, 2, 3, 4];
    let x2 = [(), (), ()];
    let e1: [uint, ..0] = [];
    let e2: [&'static str, ..0] = [];
    let e3: [(), ..0] = [];
    assert!(format!("{:?}", x) == "[1u, 2u, 3u, 4u]".to_string());
    assert!(format!("{:?}", x2) == "[(), (), ()]".to_string());
    assert!(format!("{:?}", e1) == "[]".to_string());
    assert!(format!("{:?}", e2) == "[]".to_string());
    assert!(format!("{:?}", e3) == "[]".to_string());

    let rx: &[uint, ..4] = &x;
    let rx2: &[(), ..3] = &x2;
    let re1: &[uint, ..0] = &e1;
    let re2: &[&'static str, ..0] = &e2;
    let re3: &[(), ..0] = &e3;
    assert!(format!("{:?}", rx) == "&[1u, 2u, 3u, 4u]".to_string());
    assert!(format!("{:?}", rx2) == "&[(), (), ()]".to_string());
    assert!(format!("{:?}", re1) == "&[]".to_string());
    assert!(format!("{:?}", re2) == "&[]".to_string());
    assert!(format!("{:?}", re3) == "&[]".to_string());

    let rx: &[uint] = &x;
    let rx2: &[()] = &x2;
    let re1: &[uint] = &e1;
    let re2: &[&'static str] = &e2;
    let re3: &[()] = &e3;
    assert!(format!("{:?}", rx) == "&[1u, 2u, 3u, 4u]".to_string());
    assert!(format!("{:?}", rx2) == "&[(), (), ()]".to_string());
    assert!(format!("{:?}", re1) == "&[]".to_string());
    assert!(format!("{:?}", re2) == "&[]".to_string());
    assert!(format!("{:?}", re3) == "&[]".to_string());

    // FIXME(15049) These should all work some day.
    /*let rx: Box<[uint, ..4]> = box x;
    let rx2: Box<[(), ..3]> = box x2;
    let re1: Box<[uint, ..0]> = box e1;
    let re2: Box<[&'static str, ..0]> = box e2;
    let re3: Box<[(), ..0]> = box e3;
    assert!(format!("{:?}", rx) == "box [1u, 2u, 3u, 4u]".to_string());
    assert!(format!("{:?}", rx2) == "box [(), (), ()]".to_string());
    assert!(format!("{:?}", re1) == "box []".to_string());
    assert!(format!("{:?}", re2) == "box []".to_string());
    assert!(format!("{:?}", re3) == "box []".to_string());

    let x = [1, 2, 3, 4];
    let x2 = [(), (), ()];
    let e1: [uint, ..0] = [];
    let e2: [&'static str, ..0] = [];
    let e3: [(), ..0] = [];
    let rx: Box<[uint]> = box x;
    let rx2: Box<[()]> = box x2;
    let re1: Box<[uint]> = box e1;
    let re2: Box<[&'static str]> = box e2;
    let re3: Box<[()]> = box e3;
    assert!(format!("{:?}", rx) == "box [1u, 2u, 3u, 4u]".to_string());
    assert!(format!("{:?}", rx2) == "box [(), (), ()]".to_string());
    assert!(format!("{:?}", re1) == "box []".to_string());
    assert!(format!("{:?}", re2) == "box []".to_string());
    assert!(format!("{:?}", re3) == "box []".to_string());*/
}

fn sub_expr() {
    // Test for a &[T] => &&[T] coercion in sub-expression position
    // (surpisingly, this can cause errors which are not caused by either of:
    //    `let x = vec.slice_mut(0, 2);`
    //    `foo(vec.slice_mut(0, 2));` ).
    let mut vec: Vec<int> = vec!(1, 2, 3, 4);
    let b: &mut [int] = [1, 2];
    assert!(vec.slice_mut(0, 2) == b);
}

fn index() {
    // Tests for indexing into box/& [T, ..n]
    let x: [int, ..3] = [1, 2, 3];
    let mut x: Box<[int, ..3]> = box x;
    assert!(x[0] == 1);
    assert!(x[1] == 2);
    assert!(x[2] == 3);
    x[1] = 45;
    assert!(x[0] == 1);
    assert!(x[1] == 45);
    assert!(x[2] == 3);

    let mut x: [int, ..3] = [1, 2, 3];
    let x: &mut [int, ..3] = &mut x;
    assert!(x[0] == 1);
    assert!(x[1] == 2);
    assert!(x[2] == 3);
    x[1] = 45;
    assert!(x[0] == 1);
    assert!(x[1] == 45);
    assert!(x[2] == 3);
}

pub fn main() {
    reflect();
    sub_expr();
    index();
}
