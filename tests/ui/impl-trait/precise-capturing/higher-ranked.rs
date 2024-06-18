//@ check-pass

// Show how precise captures allow us to skip capturing a higher-ranked lifetime

#![feature(lifetime_capture_rules_2024, precise_capturing)]

trait Trait<'a> {
    type Item;
}

impl Trait<'_> for () {
    type Item = Vec<()>;
}

fn hello() -> impl for<'a> Trait<'a, Item = impl IntoIterator + use<>> {}

fn main() {}
