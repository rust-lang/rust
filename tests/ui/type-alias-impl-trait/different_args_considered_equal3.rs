//! Test that we don't allow coercing an opaque type with a non-static
//! lifetime to one with a static lifetime. While `get_iter` looks like
//! it would be doing the opposite, the way we're handling projections
//! makes `Opaque<'a>` the hidden type of `Opaque<'static>`.

#![feature(type_alias_impl_trait)]

mod defining_scope {
    pub type Opaque<'a> = impl Sized;

    fn get_one<'a>(a: *mut &'a str) -> Opaque<'a> {
        a
    }
}
use defining_scope::Opaque;

fn get_iter<'a>() -> impl IntoIterator<Item = Opaque<'a>> {
    None::<Opaque<'static>>
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
