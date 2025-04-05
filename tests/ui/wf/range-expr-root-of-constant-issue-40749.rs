fn main() {
    [0; ..10];
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected type `usize`
    //~| NOTE_NONVIRAL found struct `RangeTo<{integer}>`
}
