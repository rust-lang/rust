#[link_ordinal(123)]
//~^ ERROR attribute cannot be used on
struct Foo {}

#[link_ordinal(123)]
//~^ ERROR attribute cannot be used on
fn test() {}

#[link_ordinal(42)]
//~^ ERROR attribute cannot be used on
static mut imported_val: i32 = 123;

#[link(name = "exporter")]
extern "C" {
    #[link_ordinal(13)]
    fn imported_function();

    #[link_ordinal(42)]
    static mut imported_variable: i32;
}

fn main() {}
