#![cfg_attr(target_arch = "x86", feature(raw_dylib))]

#[link_ordinal(123)]
//~^ ERROR attribute should be applied to a foreign function or static
struct Foo {}

#[link_ordinal(123)]
//~^ ERROR attribute should be applied to a foreign function or static
fn test() {}

#[link_ordinal(42)]
//~^ ERROR attribute should be applied to a foreign function or static
static mut imported_val: i32 = 123;

#[link(name = "exporter", kind = "raw-dylib")]
extern {
    #[link_ordinal(13)]
    fn imported_function();

    #[link_ordinal(42)]
    static mut imported_variable: i32;
}

fn main() {}
