#[link(name = "foo")]
extern "C" {
    #[link_ordinal(42)]
    //~^ ERROR: the `#[link_ordinal]` attribute is an experimental feature
    fn foo();
}

fn main() {}
