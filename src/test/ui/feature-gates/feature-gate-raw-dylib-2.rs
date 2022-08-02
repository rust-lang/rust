#[link(name = "foo")]
extern "C" {
    #[link_ordinal(42)]
    //~^ ERROR: the `#[link_ordinal]` attribute is an experimental feature
    fn foo();
    #[link_ordinal(5)]
    //~^ ERROR: the `#[link_ordinal]` attribute is an experimental feature
    static mut imported_variable: i32;
}

fn main() {}
