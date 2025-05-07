//@ check-pass
//@ edition: 2018

#![feature(try_blocks)]

fn main() {
    let _ = match 1 {
        1 => try {}
        _ => Ok::<(), ()>(()),
    };
}
