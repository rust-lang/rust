#![feature(coverage_attribute)]
//@ edition: 2024
//@ revisions: none some
//@[some] ignore-coverage-map

// Basic test for let-else statements.

fn let_else_semi(opt_msg: Option<&str>) {
    let Some(msg) = opt_msg else {
        return;
    };
    say(msg);
}

#[rustfmt::skip]
fn let_else_no_semi(opt_msg: Option<&str>) {
    let Some(msg) = opt_msg else {
        return
    };
    say(msg);
}

#[coverage(off)]
fn main() {
    let opt_msg = cfg_select!(
        some => Some("hello"),
        none => None,
    );
    let_else_semi(opt_msg);
    let_else_no_semi(opt_msg);
}

#[coverage(off)]
fn say(msg: &str) {
    println!("{msg}");
}
