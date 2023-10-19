// build-pass
// edition:2018

#![feature(coroutines)]

fn main() {
    let _ = static |x: u8| match x {
        y if { yield } == y + 1 => (),
        _ => (),
    };
}
