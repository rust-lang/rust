//@ only-windows
#[link(name = "foo", kind = "raw-dylib")]
extern "C" {
    #[link_ordinal(1)]
    #[link_ordinal(2)] //~ ERROR multiple `link_ordinal` attributes
    fn foo();
    #[link_ordinal(1)]
    #[link_ordinal(2)] //~ ERROR multiple `link_ordinal` attributes
    static mut imported_variable: i32;
}

fn main() {}
