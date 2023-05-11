struct Foo {
  bar1: Bar,
  bar2: Bar
}

struct Bar {
  int1: isize,
  int2: isize,
}

fn borrow_same_field_twice_mut_mut(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1;
    let _bar2 = &mut foo.bar1;  //~ ERROR cannot borrow
    use_mut(_bar1);
}
fn borrow_same_field_twice_mut_imm(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1;
    let _bar2 = &foo.bar1;  //~ ERROR cannot borrow
    use_mut(_bar1);
}
fn borrow_same_field_twice_imm_mut(foo: &mut Foo) {
    let _bar1 = &foo.bar1;
    let _bar2 = &mut foo.bar1;  //~ ERROR cannot borrow
    use_imm(_bar1);
}
fn borrow_same_field_twice_imm_imm(foo: &mut Foo) {
    let _bar1 = &foo.bar1;
    let _bar2 = &foo.bar1;
    use_imm(_bar1);
}
fn borrow_both_mut(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1;
    let _bar2 = &mut foo.bar2;
    use_mut(_bar1);
}
fn borrow_both_mut_pattern(foo: &mut Foo) {
    match *foo {
        Foo { bar1: ref mut _bar1, bar2: ref mut _bar2 } =>
        { use_mut(_bar1); use_mut(_bar2); }
    }
}
fn borrow_var_and_pattern(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1;
    match *foo {
        Foo { bar1: ref mut _bar1, bar2: _ } => {}
        //~^ ERROR cannot borrow
    }
    use_mut(_bar1);
}
fn borrow_mut_and_base_imm(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1.int1;
    let _foo1 = &foo.bar1; //~ ERROR cannot borrow
    let _foo2 = &*foo; //~ ERROR cannot borrow
    use_mut(_bar1);
}
fn borrow_mut_and_base_mut(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1.int1;
    let _foo1 = &mut foo.bar1; //~ ERROR cannot borrow
    use_mut(_bar1);
}
fn borrow_mut_and_base_mut2(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1.int1;
    let _foo2 = &mut *foo; //~ ERROR cannot borrow
    use_mut(_bar1);
}
fn borrow_imm_and_base_mut(foo: &mut Foo) {
    let _bar1 = &foo.bar1.int1;
    let _foo1 = &mut foo.bar1; //~ ERROR cannot borrow
    use_imm(_bar1);
}
fn borrow_imm_and_base_mut2(foo: &mut Foo) {
    let _bar1 = &foo.bar1.int1;
    let _foo2 = &mut *foo; //~ ERROR cannot borrow
    use_imm(_bar1);
}
fn borrow_imm_and_base_imm(foo: &mut Foo) {
    let _bar1 = &foo.bar1.int1;
    let _foo1 = &foo.bar1;
    let _foo2 = &*foo;
    use_imm(_bar1);
}
fn borrow_mut_and_imm(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1;
    let _foo1 = &foo.bar2;
    use_mut(_bar1);
}
fn borrow_mut_from_imm(foo: &Foo) {
    let _bar1 = &mut foo.bar1; //~ ERROR cannot borrow
}

fn borrow_long_path_both_mut(foo: &mut Foo) {
    let _bar1 = &mut foo.bar1.int1;
    let _foo1 = &mut foo.bar2.int2;
    use_mut(_bar1);
}
fn main() {}

fn use_mut<T>(_: &mut T) { }
fn use_imm<T>(_: &T) { }
