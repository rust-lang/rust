// Checks that we properly add implied bounds from unnormalized projections in
// inputs when typechecking functions.

//@ check-pass

trait MyTrait {
    type Assoc<'a, 'b> where 'b: 'a;
    fn do_sth(arg: Self::Assoc<'_, '_>);
    fn do_sth2(arg: Self::Assoc<'_, '_>) {}
}

struct Foo;

impl MyTrait for Foo {
    type Assoc<'a, 'b> = u32 where 'b: 'a;

    fn do_sth(_: u32) {}
    fn do_sth2(_: Self::Assoc<'static, 'static>) {}
}

fn main() {}
