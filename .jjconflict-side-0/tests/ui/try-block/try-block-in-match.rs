//@ run-pass
//@ edition: 2018

#![feature(try_blocks)]

fn main() {
    match try { } {
        Err(()) => (),
        Ok(()) => (),
    }
}
