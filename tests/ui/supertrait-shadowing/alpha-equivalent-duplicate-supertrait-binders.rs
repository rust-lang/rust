//@ check-pass
// positive test for issue - https://github.com/rust-lang/rust/issues/161547
#![feature(non_lifetime_binders)]
#![feature(supertrait_item_shadowing)]
trait E<'e> {
    type As;
}

trait F: for<'a> E<'a> + for<'b> E<'b> {}

struct G<T>
where
    T: F<As: E<'static>>,
{
    x: T,
}

fn main() {}
