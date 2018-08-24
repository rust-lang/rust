// Test that `<F as Foo<'a>>::Type: 'b`, where `trait Foo<'a> { Type:
// 'a; }`, does not require that `F: 'b`.

#![feature(rustc_attrs)]
#![allow(dead_code)]

trait SomeTrait<'a> {
    type Type: 'a;
}

impl<'a: 'c, 'c, T> SomeTrait<'a> for &'c T where T: SomeTrait<'a> {
    type Type = <T as SomeTrait<'a>>::Type;
    //          ~~~~~~~~~~~~~~~~~~~~~~~~~~
    //                       |
    // Note that this type must outlive 'a, due to the trait
    // definition.  If we fall back to OutlivesProjectionComponents
    // here, then we would require that `T:'a`, which is too strong.
}

#[rustc_error]
fn main() { } //~ ERROR compilation successful
