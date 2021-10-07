#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete

#[link(name = "foo")]
extern "C" {
    #[link_ordinal(3, 4)]
    //~^ ERROR incorrect number of arguments to `#[link_ordinal]`
    fn foo();
}

fn main() {}
