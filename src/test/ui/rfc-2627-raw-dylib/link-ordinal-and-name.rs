#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete and may cause the compiler to crash

#[link(name="foo")]
extern {
    #[link_name="foo"]
    #[link_ordinal(42)]
    //~^ ERROR cannot use `#[link_name]` with `#[link_ordinal]`
    fn foo();
}

fn main() {}
