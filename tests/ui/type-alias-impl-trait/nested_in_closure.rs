#![feature(type_alias_impl_trait)]

fn main() {
    let x = || {
        type Tait = impl Sized;
        let y: Tait = ();
        //~^ ERROR: item constrains opaque type that is not in its signature
        //~| ERROR: item constrains opaque type that is not in its signature
    };
}
