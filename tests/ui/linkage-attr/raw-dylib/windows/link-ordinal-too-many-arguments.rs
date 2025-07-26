#[link(name = "foo")]
extern "C" {
    #[link_ordinal(3, 4)]
    //~^ ERROR malformed `link_ordinal` attribute input
    //~| NOTE  expected a single argument
    fn foo();
    #[link_ordinal(3, 4)]
    //~^ ERROR malformed `link_ordinal` attribute input
    //~| NOTE  expected a single argument
    static mut imported_variable: i32;
}

fn main() {}
