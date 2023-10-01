#![feature(extern_types)]
#![feature(coverage_attribute)]
#![feature(impl_trait_in_assoc_type)]
#![warn(unused_attributes)]
#![coverage(off)]
//~^ WARN: `#[coverage]` does not propagate into items and must be applied to the contained functions directly

#[coverage(off)]
//~^ WARN: `#[coverage]` does not propagate into items and must be applied to the contained functions directly
trait Trait {
    #[coverage(off)] //~ ERROR `#[coverage]` must be applied to coverable code
    const X: u32;

    #[coverage(off)] //~ ERROR `#[coverage]` must be applied to coverable code
    type T;

    type U;
}

#[coverage(off)]
//~^ WARN: `#[coverage]` does not propagate into items and must be applied to the contained functions directly
impl Trait for () {
    const X: u32 = 0;

    #[coverage(off)] //~ ERROR `#[coverage]` must be applied to coverable code
    type T = Self;

    #[coverage(off)] //~ ERROR `#[coverage]` must be applied to coverable code
    type U = impl Trait; //~ ERROR unconstrained opaque type
}

extern "C" {
    #[coverage(off)] //~ ERROR `#[coverage]` must be applied to coverable code
    static X: u32;

    #[coverage(off)] //~ ERROR `#[coverage]` must be applied to coverable code
    type T;
}

#[coverage(off)]
fn main() {
    #[coverage(off)]
    //~^ WARN `#[coverage]` may only be applied to function definitions
    let _ = ();

    match () {
        #[coverage(off)]
        //~^ WARN `#[coverage]` may only be applied to function definitions
        () => (),
    }

    #[coverage(off)]
    //~^ WARN `#[coverage]` may only be applied to function definitions
    return ();
}
