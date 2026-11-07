#![feature(type_alias_impl_trait)]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn dyn_hoops<T: Sized>() -> dyn for<'a> Iterator<Item = impl Captures<'a>> {
    //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
    loop {}
}

pub fn main() {
    //~^ ERROR item does not constrain `Opaque::{opaque#0}`
    type Opaque = impl Sized;
    fn define() -> Opaque {
        let x: Opaque = dyn_hoops::<()>();
        x
    }
}
