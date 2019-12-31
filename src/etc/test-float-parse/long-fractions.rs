mod _common;

use _common::validate;
use std::char;

fn main() {
    for n in 0..10 {
        let digit = char::from_digit(n, 10).unwrap();
        let mut s = "0.".to_string();
        for _ in 0..400 {
            s.push(digit);
            if s.parse::<f64>().is_ok() {
                validate(&s);
            }
        }
    }
}
