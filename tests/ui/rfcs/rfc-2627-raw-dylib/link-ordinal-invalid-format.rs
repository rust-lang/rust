#[link(name = "foo")]
extern "C" {
    #[link_ordinal("JustMonika")]
    //~^ ERROR illegal ordinal format in `link_ordinal`
    fn foo();
    #[link_ordinal("JustMonika")]
    //~^ ERROR illegal ordinal format in `link_ordinal`
    static mut imported_variable: i32;
}

fn main() {}
