mod _common;

use _common::validate;
use std::u64;

fn main() {
    for exp in 19..64 {
        let power: u64 = 1 << exp;
        validate(&power.to_string());
        for offset in 1..123 {
            validate(&(power + offset).to_string());
            validate(&(power - offset).to_string());
        }
    }
    for offset in 0..123 {
        validate(&(u64::MAX - offset).to_string());
    }
}
