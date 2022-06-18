#![feature(extern_types)]
#![feature(no_coverage)]
#![feature(type_alias_impl_trait)]
#![warn(unused_attributes)]
#![no_coverage]
//~^ WARN: `#[no_coverage]` does not propagate into items and must be applied to the contained functions directly

#[no_coverage]
//~^ WARN: `#[no_coverage]` does not propagate into items and must be applied to the contained functions directly
trait Trait {
    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    const X: u32;

    #[no_coverage] //~ ERROR `#[no_coverage]` must be applied to coverable code
    type T;

    type U;
}

#[no_coverage]
//~^ WARN: `#[no_coverage]` does not propagate into items and must be applied to the contained functions directly
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
    #[no_coverage]
    //~^ WARN `#[no_coverage]` may only be applied to function definitions
    let _ = ();

    match () {
        #[no_coverage]
        //~^ WARN `#[no_coverage]` may only be applied to function definitions
        () => (),
    }

    #[no_coverage]
    //~^ WARN `#[no_coverage]` may only be applied to function definitions
    return ();
}
