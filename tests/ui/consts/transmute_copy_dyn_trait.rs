//@ run-pass
use std::mem::transmute_copy;

// Tests `transmute_copy` with an unsized dyn Trait source in const evaluation,
// covering both an exact-size transmutation and a shrinking transmutation.

trait Foo {}

impl Foo for u64 {}
impl Foo for (u64, u64) {}

const VALUE: u64 = 0x0102_0304_0506_0708;
const OBJ: &dyn Foo = &VALUE;
// Equal size case: the concrete type behind the vtable (`u64`) is exactly 8 bytes.
const EXACT: u64 = unsafe { transmute_copy::<dyn Foo, u64>(OBJ) };
const _: () = assert!(EXACT == VALUE);

const PAIR: (u64, u64) = (VALUE, 0);
const PAIR_OBJ: &dyn Foo = &PAIR;
// Shrinking case: concrete type behind the vtable is 16 bytes, `Dst` is 8.
const SHRUNK: u64 = unsafe { transmute_copy::<dyn Foo, u64>(PAIR_OBJ) };
const _: () = assert!(SHRUNK == PAIR.0);

fn main() {
    assert_eq!(EXACT, VALUE);
    assert_eq!(SHRUNK, PAIR.0);
}
