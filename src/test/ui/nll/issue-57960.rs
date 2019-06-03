// run-pass

#![allow(dead_code)]

trait Range {
    const FIRST: u8;
    const LAST: u8;
}

struct OneDigit;
impl Range for OneDigit {
    const FIRST: u8 = 0;
    const LAST: u8 = 9;
}

struct TwoDigits;
impl Range for TwoDigits {
    const FIRST: u8 = 10;
    const LAST: u8 = 99;
}

struct ThreeDigits;
impl Range for ThreeDigits {
    const FIRST: u8 = 100;
    const LAST: u8 = 255;
}

fn digits(x: u8) -> u32 {
    match x {
        OneDigit::FIRST..=OneDigit::LAST => 1,
        TwoDigits::FIRST..=TwoDigits::LAST => 2,
        ThreeDigits::FIRST..=ThreeDigits::LAST => 3,
        _ => unreachable!(),
    }
}

fn main() {
    assert_eq!(digits(100), 3);
}
