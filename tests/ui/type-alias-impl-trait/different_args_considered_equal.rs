#![feature(type_alias_impl_trait)]

pub type Opaque<'a> = impl Sized;

#[defines(Opaque)]
fn get_one<'a>(a: *mut &'a str) -> Opaque<'a> {
    a
}

#[defines(Opaque)]
fn get_iter<'a>() -> impl IntoIterator<Item = Opaque<'a>> {
    //~^ ERROR:  item does not constrain
    None::<Opaque<'static>>
}

fn main() {}
