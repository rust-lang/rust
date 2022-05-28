#![feature(extern_types)]
#![feature(no_coverage)]
#![feature(type_alias_impl_trait)]
#![warn(unused_attributes)]

trait Trait {
    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    const X: u32;

    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    type T;

    type U;
}

impl Trait for () {
    const X: u32 = 0;

    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    type T = Self;

    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    type U = impl Trait; //~ ERROR unconstrained opaque type
}

extern "C" {
    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    static X: u32;

    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    type T;
}

#[no_coverage]
fn main() {
    #[no_coverage] //~ WARN `#[no_coverage]` can only be applied at the function level, not on code directly
    let _ = ();

    match () {
        #[no_coverage] //~ WARN `#[no_coverage]` can only be applied at the function level, not on code directly
        () => (),
    }

    #[no_coverage] //~ WARN `#[no_coverage]` can only be applied at the function level, not on code directly
    return ();
}
