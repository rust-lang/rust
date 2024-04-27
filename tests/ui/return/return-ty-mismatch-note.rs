// Checks existence of a note for "a caller chooses ty for ty param" upon return ty mismatch.

fn f<T>() -> (T,) {
    (0,) //~ ERROR mismatched types
}

fn g<U, V>() -> (U, V) {
    (0, "foo")
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}

fn h() -> u8 {
    0u8
}

fn main() {
    f::<()>();
    g::<(), ()>;
    let _ = h();
}
