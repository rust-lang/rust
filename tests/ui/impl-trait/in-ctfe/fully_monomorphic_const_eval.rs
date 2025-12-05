//! This test ensures that we do look at the hidden types of
//! opaque types during const eval in order to obtain the exact type
//! of associated types.

//@ check-pass

trait MyTrait: Copy {
    const ASSOC: usize;
}

impl MyTrait for u8 {
    const ASSOC: usize = 32;
}

const fn yeet() -> impl MyTrait {
    0u8
}

const fn output<T: MyTrait>(_: T) -> usize {
    <T as MyTrait>::ASSOC
}

struct Foo<'a>(&'a ());
const NEED_REVEAL_ALL: usize = output(yeet());

fn promote_div() -> &'static usize {
    &(10 / NEED_REVEAL_ALL)
}
fn main() {}
