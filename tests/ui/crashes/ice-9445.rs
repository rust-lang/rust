const UNINIT: core::mem::MaybeUninit<core::cell::Cell<&'static ()>> = core::mem::MaybeUninit::uninit();
//~^ ERROR: a `const` item should never be interior mutable
//~| NOTE: `-D clippy::declare-interior-mutable-const` implied by `-D warnings`

fn main() {}
