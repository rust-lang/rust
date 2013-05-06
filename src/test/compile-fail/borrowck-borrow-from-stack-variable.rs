// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
  bar1: Bar,
  bar2: Bar
}

struct Bar {
  int1: int,
  int2: int,
}

fn make_foo() -> Foo { fail!() }

fn borrow_same_field_twice_mut_mut() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1;
    let _bar2 = &mut foo.bar1;  //~ ERROR cannot borrow
    *bar1;
}

fn borrow_same_field_twice_mut_imm() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1;
    let _bar2 = &foo.bar1;  //~ ERROR cannot borrow
    *bar1;
}

fn borrow_same_field_twice_imm_mut() {
    let mut foo = make_foo();
    let bar1 = &foo.bar1;
    let _bar2 = &mut foo.bar1;  //~ ERROR cannot borrow
    *bar1;
}

fn borrow_same_field_twice_imm_imm() {
    let mut foo = make_foo();
    let bar1 = &foo.bar1;
    let _bar2 = &foo.bar1;
    *bar1;
}

fn borrow_both_mut() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1;
    let _bar2 = &mut foo.bar2;
    *bar1;
}

fn borrow_both_mut_pattern() {
    let mut foo = make_foo();
    match foo {
        Foo { bar1: ref mut _bar1, bar2: ref mut _bar2 } => {}
    }
}

fn borrow_var_and_pattern() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1;
    match foo {
        Foo { bar1: ref mut _bar1, bar2: _ } => {} //
        //~^ ERROR cannot borrow
    }
    *bar1;
}

fn borrow_mut_and_base_imm() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1.int1;
    let _foo1 = &foo.bar1; //~ ERROR cannot borrow
    let _foo2 = &foo; //~ ERROR cannot borrow
    *bar1;
}

fn borrow_mut_and_base_mut() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1.int1;
    let _foo1 = &mut foo.bar1; //~ ERROR cannot borrow
    *bar1;
}

fn borrow_mut_and_base_mut2() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1.int1;
    let _foo2 = &mut foo; //~ ERROR cannot borrow
    *bar1;
}

fn borrow_imm_and_base_mut() {
    let mut foo = make_foo();
    let bar1 = &foo.bar1.int1;
    let _foo1 = &mut foo.bar1; //~ ERROR cannot borrow
    *bar1;
}

fn borrow_imm_and_base_mut2() {
    let mut foo = make_foo();
    let bar1 = &foo.bar1.int1;
    let _foo2 = &mut foo; //~ ERROR cannot borrow
    *bar1;
}

fn borrow_imm_and_base_imm() {
    let mut foo = make_foo();
    let bar1 = &foo.bar1.int1;
    let _foo1 = &foo.bar1;
    let _foo2 = &foo;
    *bar1;
}

fn borrow_mut_and_imm() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1;
    let _foo1 = &foo.bar2;
    *bar1;
}

fn borrow_mut_from_imm() {
    let foo = make_foo();
    let bar1 = &mut foo.bar1; //~ ERROR cannot borrow
    *bar1;
}

fn borrow_long_path_both_mut() {
    let mut foo = make_foo();
    let bar1 = &mut foo.bar1.int1;
    let _foo1 = &mut foo.bar2.int2;
    *bar1;
}

fn main() {}
