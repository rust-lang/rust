#![feature(type_alias_impl_trait)]

type Tait<T> = impl Sized;

#[define_opaque(Tait)]
fn foo<T, U>() -> Tait<T> {
    if false {
        if { return } {
            let y: Tait<U> = 1i32;
            //~^ ERROR concrete type differs from previous defining opaque type use
        }
    }
    let x: Tait<T> = ();
    x
}

fn main() {}
