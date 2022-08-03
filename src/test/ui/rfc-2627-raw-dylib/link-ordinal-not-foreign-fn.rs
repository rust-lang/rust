#![feature(raw_dylib)]
//~^ WARN the feature `raw_dylib` is incomplete

#[link_ordinal(123)]
//~^ ERROR attribute should be applied to a foreign function
struct Foo {}

#[link_ordinal(123)]
//~^ ERROR attribute should be applied to a foreign function
fn test() {}

#[link(name = "exporter", kind = "raw-dylib")]
extern {
    #[link_ordinal(13)]
    fn imported_function();
}

fn main() {}
