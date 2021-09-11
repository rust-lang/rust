// only-windows-msvc
#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete

#[link(name = "foo", kind = "raw-dylib")]
extern "C" {
    #[link_ordinal(1)]
    #[link_ordinal(2)]
    //~^ ERROR multiple `link_ordinal` attributes on a single definition
    fn foo();
}

fn main() {}
