// aux-build:roman-numerals.rs
// ignore-stage1

#![feature(plugin)]
#![plugin(roman_numerals)]

pub fn main() {
    assert_eq!(rn!(MMXV), 2015);
    assert_eq!(rn!(MCMXCIX), 1999);
    assert_eq!(rn!(XXV), 25);
    assert_eq!(rn!(MDCLXVI), 1666);
    assert_eq!(rn!(MMMDCCCLXXXVIII), 3888);
    assert_eq!(rn!(MMXIV), 2014);
}
