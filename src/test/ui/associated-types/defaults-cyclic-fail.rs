#![feature(associated_type_defaults)]

// Having a cycle in assoc. type defaults is okay...
trait Tr {
    type A = Self::B;
    type B = Self::A;
}

// ...but is an error in any impl that doesn't override at least one of the defaults
impl Tr for () {}
//~^ ERROR overflow evaluating the requirement

// As soon as at least one is redefined, it works:
impl Tr for u8 {
    type A = u8;
}

impl Tr for u32 {
    type A = ();
    type B = u8;
}

// ...but only if this actually breaks the cycle
impl Tr for bool {
//~^ ERROR overflow evaluating the requirement
    type A = Box<Self::B>;
    //~^ ERROR overflow evaluating the requirement
}
// (the error is shown twice for some reason)

fn main() {
    // Check that the overridden type propagates to the other
    let _a: <u8 as Tr>::A = 0u8;
    let _b: <u8 as Tr>::B = 0u8;
}
