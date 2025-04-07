#![feature(type_alias_impl_trait)]

type Tait<'a> = impl Sized + 'a;

#[define_opaque(Tait)]
fn foo<'a, 'b>() -> Tait<'a> {
    if false {
        if { return } {
            let y: Tait<'b> = 1i32;
            //~^ ERROR concrete type differs from previous defining opaque type use
        }
    }
    let x: Tait<'a> = ();
    x
}

fn main() {}
