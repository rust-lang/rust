#![feature(coverage_attribute)]
//@ edition: 2021
//@ min-llvm-version: 19
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

#[derive(Clone, Copy)]
enum Pat {
    A(Option<i32>),
    B(i32),
    C(i32),
}

fn single_nested_pattern(pat: Pat) {
    if let Pat::A(Some(_)) = pat {
        say("matched");
    }
}

fn simple_or_pattern(pat: Pat) {
    if let Pat::B(_) | Pat::C(_) = pat {
        say("matched");
    }
}

fn simple_joint_pattern(pat: (Pat, Pat)) {
    if let (Pat::A(Some(_)), Pat::B(_)) = pat {
        say("matched");
    }
}

fn joint_pattern_with_or(pat: (Pat, Pat)) {
    if let (Pat::A(Some(_)) | Pat::C(_), Pat::B(_)) = pat {
        say("matched");
    }
}

fn joint_or_patterns(pat: (Pat, Pat)) {
    if let (Pat::A(Some(_)) | Pat::C(_), Pat::B(_) | Pat::C(_)) = pat {
        say("matched");
    }

    // Try to use the matched value
    if let (Pat::A(Some(a)) | Pat::C(a), Pat::B(b) | Pat::C(b)) = pat {
        say(&format!("matched {a} and {b}"));
    }
}

fn let_else(value: Pat) {
    let Pat::A(Some(_)) = value else { return };
    say("matched");
}

#[coverage(off)]
fn main() {
    single_nested_pattern(Pat::A(Some(5)));
    single_nested_pattern(Pat::B(5));

    simple_or_pattern(Pat::A(None));
    simple_or_pattern(Pat::C(3));

    simple_joint_pattern((Pat::A(Some(1)), Pat::B(2)));
    simple_joint_pattern((Pat::A(Some(1)), Pat::C(2)));
    simple_joint_pattern((Pat::B(1), Pat::B(2)));

    joint_pattern_with_or((Pat::A(Some(1)), Pat::B(2)));
    joint_pattern_with_or((Pat::B(1), Pat::C(2)));
    joint_pattern_with_or((Pat::B(1), Pat::B(2)));
    joint_pattern_with_or((Pat::C(1), Pat::B(2)));

    joint_or_patterns((Pat::A(Some(1)), Pat::B(2)));
    joint_or_patterns((Pat::B(1), Pat::C(2)));
    joint_or_patterns((Pat::B(1), Pat::B(2)));
    joint_or_patterns((Pat::C(1), Pat::B(2)));

    let_else(Pat::A(Some(5)));
    let_else(Pat::B(3));
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}
