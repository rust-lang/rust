// https://github.com/rust-lang/rust/issues/9047
//@ run-pass
#![allow(unused_mut)]
#![allow(unused_variables)]
fn decode() -> String {
    'outer: loop {
        let mut ch_start: usize;
        break 'outer;
    }
    "".to_string()
}

pub fn main() {
    println!("{}", decode());
}
