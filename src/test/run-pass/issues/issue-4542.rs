// run-pass
// pretty-expanded FIXME #23616
// ignore-cloudabi no std::env

use std::env;

pub fn main() {
    for arg in env::args() {
        match arg.clone() {
            _s => { }
        }
    }
}
