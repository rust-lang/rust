//@ check-pass
//@ edition: 2024

// Show how precise captures allow us to skip capturing a higher-ranked lifetime

trait Trait<'a> {
    type Item;
}

impl Trait<'_> for () {
    type Item = Vec<()>;
}

fn hello() -> impl for<'a> Trait<'a, Item = impl IntoIterator + use<>> {}

fn main() {}
