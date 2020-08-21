#![feature(associated_type_defaults)]

// A more complex version of `defaults-cyclic-fail-1.rs`, with non-trivial defaults.

// Having a cycle in assoc. type defaults is okay...
trait Tr {
    type A = Vec<Self::B>;
    type B = Box<Self::A>;
}

// ...but is an error in any impl that doesn't override at least one of the defaults
impl Tr for () {}
//~^ ERROR type mismatch resolving `<() as Tr>::B == _`

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
    //~^ ERROR type mismatch resolving `<bool as Tr>::B == _`
    type A = Box<Self::B>;
    //~^ ERROR type mismatch resolving `<bool as Tr>::B == _`
}
// (the error is shown twice for some reason)

impl Tr for usize {
    //~^ ERROR type mismatch resolving `<usize as Tr>::B == _`
    type B = &'static Self::A;
    //~^ ERROR type mismatch resolving `<usize as Tr>::A == _`
}

fn main() {
    // We don't check that the types project correctly because the cycle errors stop compilation
    // before `main` is type-checked.
    // `defaults-cyclic-pass-2.rs` does this.
}
