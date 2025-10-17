#![feature(type_alias_impl_trait)]

pub type Opaque<'a> = impl Sized;

#[define_opaque(Opaque)]
fn get_one<'a>(a: *mut &'a str) -> impl IntoIterator<Item = Opaque<'a>> {
    if a.is_null() {
        Some(a)
    } else {
        None::<Opaque<'static>>
        //~^ ERROR expected generic lifetime parameter, found `'static`
    }
}

fn main() {}
