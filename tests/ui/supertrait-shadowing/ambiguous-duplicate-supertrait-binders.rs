// negative test for issue - https://github.com/rust-lang/rust/issues/161547
#![feature(non_lifetime_binders)]
#![feature(supertrait_item_shadowing)]
trait E<'e> {
    type As;
}

trait F: for<F> E + for<'e> E<'e> {} //~ ERROR missing lifetime specifier

struct G<T>
where
    T: F<As: E<'static>>, //~ ERROR ambiguous associated type `As` in bounds of `F`
{
    X: T,
}

fn main() {}
