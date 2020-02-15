#![feature(associated_type_defaults)]

// A more complex version of `defaults-cyclic-fail-1.rs`, with non-trivial defaults.

// Having a cycle in assoc. type defaults is okay...
trait Tr {
    type A = Vec<Self::B>;
    type B = Box<Self::A>;
}

// ...but is an error in any impl that doesn't override at least one of the defaults
impl Tr for () {}
//~^ ERROR overflow evaluating the requirement

// As soon as at least one is redefined, it works:
impl Tr for u8 {
    type A = u8;
}

impl Tr for u16 {
    type B = ();
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

impl Tr for usize {
//~^ ERROR overflow evaluating the requirement
    type B = &'static Self::A;
    //~^ ERROR overflow evaluating the requirement
}

fn main() {
    // We don't check that the types project correctly because the cycle errors stop compilation
    // before `main` is type-checked.
    // `defaults-cyclic-pass-2.rs` does this.
}
