#[link(name = "foo")]
extern "C" {
    #[link_ordinal(3)]
    //~^ ERROR `#[link_ordinal]` is only supported if link kind is `raw-dylib`
    fn foo();
}

#[link(name = "bar", kind = "static")]
extern "C" {
    #[link_ordinal(3)]
    //~^ ERROR `#[link_ordinal]` is only supported if link kind is `raw-dylib`
    fn bar();
}

fn main() {}
