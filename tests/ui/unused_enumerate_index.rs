// run-rustfix
#![allow(unused)]
#![warn(clippy::unused_enumerate_index)]

fn main() {
    let v = [1, 2, 3];
    for (_, x) in v.iter().enumerate() {
        print!("{x}");
    }
}
