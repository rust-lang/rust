#![feature(extern_types)]
#![feature(coverage_attribute)]
#![feature(impl_trait_in_assoc_type)]
#![warn(unused_attributes)]
#![coverage(off)]

#[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
trait Trait {
    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    const X: u32;

    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    type T;

    type U;
}

#[coverage(off)]
impl Trait for () {
    const X: u32 = 0;

    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    type T = Self;

    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    type U = impl Trait; //~ ERROR unconstrained opaque type
}

extern "C" {
    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    static X: u32;

    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    type T;
}

#[coverage(off)]
fn main() {
    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    let _ = ();

    match () {
        #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
        () => (),
    }

    #[coverage(off)] //~ ERROR attribute should be applied to a function definition or closure
    return ();
}
