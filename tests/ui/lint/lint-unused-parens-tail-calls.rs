#![deny(unused_parens)]
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn target() -> bool {
    true
}

fn if_cond() -> bool {
    if (become target()) { true } else { false }
}

fn match_scrutinee() -> bool {
    match (become target()) {}
}

fn double_if() -> bool {
    if ((become target())) {
        //~^ ERROR unnecessary parentheses around `if` condition
        true
    } else {
        false
    }
}

fn double_match() -> bool {
    match ((become target())) {}
    //~^ ERROR unnecessary parentheses around `match` scrutinee expression
}
fn main() {}
