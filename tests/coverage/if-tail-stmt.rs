#![feature(coverage_attribute)]
//@ edition: 2024
//@ revisions: no yes
//@[yes] ignore-coverage-map

// A variety of simple `if` expressions, in which the then/else blocks end with
// a semicolon. Contrast with `if-tail-expr.rs`.

fn if_true(cond: bool, other: bool) {
    say("hello");

    if cond {
        say("true");
    }

    if cond {
        say("true");
    } else {
        say("false");
    }

    if cond {
        say("cond");
    } else if other {
        say("other");
    } else {
        say("neither");
    }

    say("goodbye");
}

#[coverage(off)]
fn main() {
    let cond = cfg_select!(
        no => false,
        yes => true,
    );
    if_true(cond, !cond);
}

#[coverage(off)]
fn say(msg: &str) {
    println!("{msg}");
}
