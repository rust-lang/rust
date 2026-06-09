//@ run-pass

use std::env;

pub fn main() {
    for arg in env::args() {
        match arg.clone() {
            _s => { }
        }
    }
}
