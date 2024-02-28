#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait Other<U: ?Sized> {}

impl<U: ?Sized> Other<U> for U {}

#[rustfmt::skip]
fn foo<U: ?Sized>()
where
    for<T> T: Other<U> {}

fn bar() {
    foo::<_>();
    //~^ ERROR trait `Other<_>` is not implemented for `T`
}

fn main() {}
