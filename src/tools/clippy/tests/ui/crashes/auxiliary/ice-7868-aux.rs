fn zero() {
    unsafe { 0 };
    //~^ ERROR: unsafe block missing a safety comment
    //~| NOTE: `-D clippy::undocumented-unsafe-blocks` implied by `-D warnings`
}
