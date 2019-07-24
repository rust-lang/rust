// Here we test all kinds of range patterns in terms of parsing / recovery.
// We want to ensure that:
// 1. Things parse as they should.
// 2. Or at least we have parser recovery if they don't.

#![feature(exclusive_range_pattern)]
#![deny(ellipsis_inclusive_range_patterns)]

fn main() {}

const X: u8 = 0;
const Y: u8 = 3;

fn exclusive_from_to() {
    if let 0..3 = 0 {} // OK.
    if let 0..Y = 0 {} // OK.
    if let X..3 = 0 {} // OK.
    if let X..Y = 0 {} // OK.
    if let true..Y = 0 {} //~ ERROR only char and numeric types
    if let X..true = 0 {} //~ ERROR only char and numeric types
    if let .0..Y = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    if let X.. .0 = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
}

fn inclusive_from_to() {
    if let 0..=3 = 0 {} // OK.
    if let 0..=Y = 0 {} // OK.
    if let X..=3 = 0 {} // OK.
    if let X..=Y = 0 {} // OK.
    if let true..=Y = 0 {} //~ ERROR only char and numeric types
    if let X..=true = 0 {} //~ ERROR only char and numeric types
    if let .0..=Y = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    if let X..=.0 = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
}

fn inclusive2_from_to() {
    if let 0...3 = 0 {} //~ ERROR `...` range patterns are deprecated
    if let 0...Y = 0 {} //~ ERROR `...` range patterns are deprecated
    if let X...3 = 0 {} //~ ERROR `...` range patterns are deprecated
    if let X...Y = 0 {} //~ ERROR `...` range patterns are deprecated
    if let true...Y = 0 {} //~ ERROR only char and numeric types
    //~^ ERROR `...` range patterns are deprecated
    if let X...true = 0 {} //~ ERROR only char and numeric types
    //~^ ERROR `...` range patterns are deprecated
    if let .0...Y = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    //~| ERROR `...` range patterns are deprecated
    if let X... .0 = 0 {} //~ ERROR mismatched types
    //~^ ERROR float literals must have an integer part
    //~| ERROR `...` range patterns are deprecated
}

fn exclusive_from() {
    if let 0.. = 0 {} //~ ERROR `X..` range patterns are not supported
    if let X.. = 0 {} //~ ERROR `X..` range patterns are not supported
    if let true.. = 0 {} //~ ERROR `X..` range patterns are not supported
    //~^ ERROR only char and numeric types
    if let .0.. = 0 {} //~ ERROR `X..` range patterns are not supported
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive_from() {
    if let 0..= = 0 {} //~ ERROR `X..=` range patterns are not supported
    if let X..= = 0 {} //~ ERROR `X..=` range patterns are not supported
    if let true..= = 0 {} //~ ERROR `X..=` range patterns are not supported
    //~| ERROR only char and numeric types
    if let .0..= = 0 {} //~ ERROR `X..=` range patterns are not supported
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive2_from() {
    if let 0... = 0 {} //~ ERROR `X...` range patterns are not supported
    //~^ ERROR `...` range patterns are deprecated
    if let X... = 0 {} //~ ERROR `X...` range patterns are not supported
    //~^ ERROR `...` range patterns are deprecated
    if let true... = 0 {} //~ ERROR `X...` range patterns are not supported
    //~^ ERROR `...` range patterns are deprecated
    //~| ERROR only char and numeric types
    if let .0... = 0 {} //~ ERROR `X...` range patterns are not supported
    //~^ ERROR float literals must have an integer part
    //~| ERROR `...` range patterns are deprecated
    //~| ERROR mismatched types
}

fn exclusive_to() {
    if let ..0 = 0 {} //~ ERROR `..X` range patterns are not supported
    if let ..Y = 0 {} //~ ERROR `..X` range patterns are not supported
    if let ..true = 0 {} //~ ERROR `..X` range patterns are not supported
    //~| ERROR only char and numeric types
    if let .. .0 = 0 {} //~ ERROR `..X` range patterns are not supported
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive_to() {
    if let ..=3 = 0 {} //~ ERROR `..=X` range patterns are not supported
    if let ..=Y = 0 {} //~ ERROR `..=X` range patterns are not supported
    if let ..=true = 0 {} //~ ERROR `..=X` range patterns are not supported
    //~| ERROR only char and numeric types
    if let ..=.0 = 0 {} //~ ERROR `..=X` range patterns are not supported
    //~^ ERROR float literals must have an integer part
    //~| ERROR mismatched types
}

fn inclusive2_to() {
    if let ...3 = 0 {} //~ ERROR `...X` range patterns are not supported
    //~^ ERROR `...` range patterns are deprecated
    if let ...Y = 0 {} //~ ERROR `...X` range patterns are not supported
    //~^ ERROR `...` range patterns are deprecated
    if let ...true = 0 {} //~ ERROR `...X` range patterns are not supported
    //~^ ERROR `...` range patterns are deprecated
    //~| ERROR only char and numeric types
    if let ....3 = 0 {} //~ ERROR `...X` range patterns are not supported
    //~^ ERROR float literals must have an integer part
    //~| ERROR `...` range patterns are deprecated
    //~| ERROR mismatched types
}
