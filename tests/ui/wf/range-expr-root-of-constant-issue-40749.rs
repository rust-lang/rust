fn main() {
    [0; ..10];
    //~^ ERROR mismatched types
    //~| NOTE expected type `usize`
    //~| NOTE found struct `RangeTo<{integer}>`
    //~| NOTE expected `usize`, found `RangeTo<{integer}>
}
