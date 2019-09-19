#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete and may cause the compiler to crash

#[link(name="foo")]
extern {
    #[link_ordinal("JustMonika")]
    //~^ ERROR illegal ordinal format in `link_ordinal`
    fn foo();
}

fn main() {}
