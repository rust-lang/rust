// only-x86
#[link(name = "foo")]
extern "C" {
    #[link_ordinal(42)]
    //~^ ERROR: `#[link_ordinal]` is unstable on x86
    fn foo();
    #[link_ordinal(5)]
    //~^ ERROR: `#[link_ordinal]` is unstable on x86
    static mut imported_variable: i32;
}

fn main() {}
