fn main() {
    &panic!()
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected unit type `()`
    //~| NOTE_NONVIRAL found reference `&_`
    //~| NOTE_NONVIRAL expected `()`, found `&_`
}
