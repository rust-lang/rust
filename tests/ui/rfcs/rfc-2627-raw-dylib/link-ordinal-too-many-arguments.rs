#[link(name = "foo")]
extern "C" {
    #[link_ordinal(3, 4)]
    //~^ ERROR incorrect number of arguments to `#[link_ordinal]`
    fn foo();
    #[link_ordinal(3, 4)]
    //~^ ERROR incorrect number of arguments to `#[link_ordinal]`
    static mut imported_variable: i32;
}

fn main() {}
