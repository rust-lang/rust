// Checks that we properly add implied bounds from unnormalized projections in
// inputs when typechecking functions.

// check-pass

#![feature(generic_associated_types)]

trait MyTrait {
    type Assoc<'a, 'b> where 'b: 'a;
    fn do_sth(arg: Self::Assoc<'_, '_>);
}

struct A;
struct B;
struct C;

impl MyTrait for A {
    type Assoc<'a, 'b> where 'b: 'a = u32;
    fn do_sth(_: u32) {}
}
impl MyTrait for B {
    type Assoc<'a, 'b> where 'b: 'a = u32;
    fn do_sth(_: Self::Assoc<'_, '_>) {}
}
impl MyTrait for C {
    type Assoc<'a, 'b> where 'b: 'a = u32;
    fn do_sth(_: Self::Assoc<'static, 'static>) {}
}

fn main () {}
