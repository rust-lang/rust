// only-windows
#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete

#[link(name = "foo", kind = "raw-dylib")]
extern "C" {
    #[link_ordinal(1)] //~ ERROR multiple `link_ordinal` attributes
    #[link_ordinal(2)]
    fn foo();
}

fn main() {}
