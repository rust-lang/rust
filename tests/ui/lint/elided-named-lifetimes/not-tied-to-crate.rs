#![allow(elided_named_lifetimes)]

#[warn(elided_named_lifetimes)]
mod foo {
    fn bar(x: &'static u8) -> &u8 {
        //~^ WARNING elided lifetime has a name
        x
    }

    #[deny(elided_named_lifetimes)]
    fn baz(x: &'static u8) -> &u8 {
        //~^ ERROR elided lifetime has a name
        x
    }
}

fn main() {}
