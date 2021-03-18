#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete

#[link(name = "foo")]
extern "C" {
    #[link_ordinal(18446744073709551616)]
    //~^ ERROR ordinal value in `link_ordinal` is too large: `18446744073709551616`
    fn foo();
}

fn main() {}
