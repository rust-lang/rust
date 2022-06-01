// Validation makes this fail in the wrong place
// Make sure we find these even with many checks disabled.
// compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

fn main() {
    let i = unsafe { std::mem::MaybeUninit::<i32>::uninit().assume_init() };
    let _x = i + 0; //~ ERROR this operation requires initialized memory
}
