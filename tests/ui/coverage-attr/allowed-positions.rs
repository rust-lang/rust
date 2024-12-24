//! Tests where the `#[coverage(..)]` attribute can and cannot be used.

//@ reference: attributes.coverage.allowed-positions

#![feature(coverage_attribute)]
#![feature(extern_types)]
#![feature(impl_trait_in_assoc_type)]
#![warn(unused_attributes)]
#![coverage(off)]

#[coverage(off)] //~ ERROR [E0788]
trait Trait {
    #[coverage(off)] //~ ERROR [E0788]
    const X: u32;

    #[coverage(off)] //~ ERROR [E0788]
    type T;

    type U;

    #[coverage(off)] //~ ERROR [E0788]
    fn f(&self);

    #[coverage(off)] //~ ERROR [E0788]
    fn g();
}

#[coverage(off)]
impl Trait for () {
    const X: u32 = 0;

    #[coverage(off)] //~ ERROR [E0788]
    type T = Self;

    #[coverage(off)] //~ ERROR [E0788]
    type U = impl Trait; //~ ERROR unconstrained opaque type

    fn f(&self) {}
    fn g() {}
}

extern "C" {
    #[coverage(off)] //~ ERROR [E0788]
    static X: u32;

    #[coverage(off)] //~ ERROR [E0788]
    type T;
}

#[coverage(off)]
fn main() {
    #[coverage(off)] //~ ERROR [E0788]
    let _ = ();

    match () {
        #[coverage(off)] //~ ERROR [E0788]
        () => (),
    }

    #[coverage(off)] //~ ERROR [E0788]
    return ();
}
