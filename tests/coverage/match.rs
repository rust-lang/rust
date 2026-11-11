#![feature(coverage_attribute)]
//@ edition: 2024

// Basic test for `match` expressions with various kinds of arms and guards.

fn match_expr(x: Option<u32>, cond: bool) {
    match x {
        Some(0) => say("zero"),
        Some(1) => {
            // (block with a trailing expression)
            say("one")
        }
        Some(2) => {
            say("two");
        }
        Some(3) if cond => {
            say("three-cond");
        }
        Some(3) => say("three"),
        Some(other) if other == 4 => {
            say("four");
        }
        Some(other) => say("other"),
        None => say("none"),
    }
}

#[coverage(off)]
fn main() {
    for i in 0..=5 {
        for _ in 0..i {
            match_expr(Some(i), false);
        }
    }
    match_expr(None, true);
}

#[coverage(off)]
fn say(msg: &str) {
    println!("{msg}");
}
