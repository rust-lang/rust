#[derive(Copy, Clone)]
struct Foo {
  bar1: Bar,
  bar2: Bar
}

#[derive(Copy, Clone)]
struct Bar {
  int1: isize,
  int2: isize,
}

fn make_foo() -> Foo { panic!() }

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
