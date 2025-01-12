#![feature(never_patterns)]
#![feature(if_let_guard)]
#![allow(incomplete_features)]

fn split_last(_: &()) -> Option<(&i32, &i32)> {
    None
}

fn assign_twice() {
    loop {
        match () {
            (!| //~ ERROR: mismatched types
            !) if let _ = split_last(&()) => {} //~ ERROR a never pattern is always unreachable
            //~^ ERROR: mismatched types
            //~^^ WARNING: irrefutable `if let` guard pattern [irrefutable_let_patterns]
            _ => {}
        }
    }
}

fn main() {}
