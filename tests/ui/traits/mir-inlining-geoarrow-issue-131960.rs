//@ revisions: current next
//@ compile-flags: -Copt-level=3
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/131960>.
// Used to overflow

pub trait One: Sized {
    type ItemOne;
    type ToTwo: Two<ItemTwo = Self::ItemOne>;
}
pub trait Two {
    type ItemTwo;
    type ToOne: One<ItemOne = Self::ItemTwo>;
}

trait OneExt {
    fn one_ext() {}
}
impl<T, C: One<ItemOne = T>> OneExt for C {}
// Using this instead makes it compile
// impl<C: One> OneExt for C {}

#[allow(unconditional_recursion)]
fn recurse<C: One>() {
    C::one_ext();
    recurse::<<C::ToTwo as Two>::ToOne>();
}

// This works fine
// #[allow(unconditional_recursion)]
// fn recurse<C: One>() {
//     fn require_one_ext<C: OneExt>() {}
//     require_one_ext::<C>();
//     recurse::<<C::ToTwo as Two>::ToOne>();
// }

// This function is necessary to reproduce the bug.
pub fn call_recurse<C: One>() {
    recurse::<C>();
}

fn main() {}
