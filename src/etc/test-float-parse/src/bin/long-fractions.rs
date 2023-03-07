use std::char;
use test_float_parse::validate;

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
