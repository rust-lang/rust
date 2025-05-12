//@ only-windows
#[link(name = "foo", kind = "raw-dylib")]
extern "C" {
    #[link_ordinal(1)] //~ ERROR multiple `link_ordinal` attributes
    #[link_ordinal(2)]
    fn foo();
    #[link_ordinal(1)] //~ ERROR multiple `link_ordinal` attributes
    #[link_ordinal(2)]
    static mut imported_variable: i32;
}

fn main() {}
