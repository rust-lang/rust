#![feature(type_alias_impl_trait)]

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn dyn_hoops<T: Sized>() -> dyn for<'a> Iterator<Item = impl Captures<'a>> {
    //~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from `dyn` type
    //~| ERROR return type cannot have an unboxed trait object
    loop {}
}

pub fn main() {
    //~^ ERROR item does not constrain `Opaque::{opaque#0}`, but has it in its signature
    type Opaque = impl Sized;
    fn define() -> Opaque {
        //~^ ERROR the size for values of type `(dyn Iterator<Item = impl Captures<'_>> + 'static)`
        let x: Opaque = dyn_hoops::<()>();
        //~^ ERROR the size for values of type `(dyn Iterator<Item = impl Captures<'_>> + 'static)`
        x
    }
}
