// check-fail
// known-bug

// This should pass, but unnormalized input args aren't treated as implied.

#![feature(generic_associated_types)]

trait MyTrait {
    type Assoc<'a, 'b> where 'b: 'a;
    fn do_sth(arg: Self::Assoc<'_, '_>);
}

struct Foo;

impl MyTrait for Foo {
    type Assoc<'a, 'b> where 'b: 'a = u32;

    fn do_sth(_: u32) {}
    // fn do_sth(_: Self::Assoc<'static, 'static>) {}
    // fn do_sth(_: Self::Assoc<'_, '_>) {}
}

fn main() {}
