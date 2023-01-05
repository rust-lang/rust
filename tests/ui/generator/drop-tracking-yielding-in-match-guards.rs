// build-pass
// edition:2018
// compile-flags: -Zdrop-tracking

#![feature(generators)]

fn main() {
    let _ = static |x: u8| match x {
        y if { yield } == y + 1 => (),
        _ => (),
    };
}
