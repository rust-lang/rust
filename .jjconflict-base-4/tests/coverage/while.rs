#![feature(coverage_attribute)]
//@ edition: 2024

// Basic test for `while` expressions.

fn while_with_tail_expr() {
    let mut x = 5;
    while x > 0 {
        x -= 1;
        say("decreased x")
    }
    say("goodbye");
}

fn while_with_tail_stmt() {
    let mut x = 5;
    while x > 0 {
        x -= 1;
        say("decreased x");
    }
    say("goodbye");
}

#[coverage(off)]
fn main() {
    while_with_tail_expr();
    while_with_tail_stmt();
}

#[coverage(off)]
fn say(msg: &str) {
    println!("{msg}");
}
