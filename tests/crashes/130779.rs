//@ known-bug: #130779
#![feature(never_patterns)]

enum E { A }

fn main() {
    match E::A {
        ! |
        if true => {}
    }
}
