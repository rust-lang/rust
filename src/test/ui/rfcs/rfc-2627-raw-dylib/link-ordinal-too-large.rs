#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete

#[link(name = "foo")]
extern "C" {
    #[link_ordinal(72436)]
    //~^ ERROR ordinal value in `link_ordinal` is too large: `72436`
    fn foo();
}

fn main() {}
