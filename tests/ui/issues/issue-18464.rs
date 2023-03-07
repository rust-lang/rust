// run-pass
#![deny(dead_code)]

const LOW_RANGE: char = '0';
const HIGH_RANGE: char = '9';

fn main() {
    match '5' {
        LOW_RANGE..=HIGH_RANGE => (),
        _ => ()
    };
}
