// Validation makes this fail in the wrong place
// Make sure we find these even with many checks disabled.
// compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

#[inline(never)]
fn dont_optimize<T>(x: T) -> T { x }

fn main() {
    let b = unsafe { std::mem::transmute::<u8, bool>(2) };
    let _x = b == dont_optimize(true); //~ ERROR interpreting an invalid 8-bit value as a bool: 0x02
}
