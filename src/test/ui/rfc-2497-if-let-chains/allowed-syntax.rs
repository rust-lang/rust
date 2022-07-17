// check-pass

#![allow(irrefutable_let_patterns)]

use std::ops::Range;

fn _if() {
    if let 0 = 1 {}

    if true && let 0 = 1 {}

    if let 0 = 1 && true {}

    if let Range { start: _, end: _ } = (true..true) && false {}

    if let 1 = 1 && let true = { true } && false {
    }
}

fn _while() {
    while let 0 = 1 {}

    while true && let 0 = 1 {}

    while let 0 = 1 && true {}

    while let Range { start: _, end: _ } = (true..true) && false {}
}

fn main() {}
