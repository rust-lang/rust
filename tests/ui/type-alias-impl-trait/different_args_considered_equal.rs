#![feature(type_alias_impl_trait)]

pub type Opaque<'a> = impl Sized;

#[define_opaque(Opaque)]
fn get_one<'a>(a: *mut &'a str) -> Opaque<'a> {
    a
}

#[define_opaque(Opaque)]
fn get_iter<'a>() -> impl IntoIterator<Item = Opaque<'a>> {
    //~^ ERROR item does not constrain `Opaque::{opaque#0}`
    None::<Opaque<'static>>
}

fn main() {}
