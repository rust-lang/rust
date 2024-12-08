// Issue 52985: user code provides no use case that allows a type alias `impl Trait`
// We now emit a 'unconstrained opaque type' error

#![feature(type_alias_impl_trait)]

mod foo {
    pub type Foo = impl Copy;

    // make compiler happy about using 'Foo'
    pub fn bar(x: Foo) -> Foo {
        //~^ ERROR: item does not constrain `Foo::{opaque#0}`
        x
    }
}

fn main() {
    unsafe {
        let _: foo::Foo = std::mem::transmute(0u8);
    }
}
