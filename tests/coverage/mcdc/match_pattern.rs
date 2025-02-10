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
    match pat {
        Pat::A(Some(_)) => say("matched A::Some"),
        Pat::A(None) => say("matched A::None"),
        Pat::B(_) => say("matched B"),
        Pat::C(_) => say("matched C"),
    }
}

fn simple_or_pattern(pat: Pat) {
    match pat {
        Pat::B(_) | Pat::C(_) => say("matched B or C"),
        _ => say("matched A"),
    }
}

fn simple_joint_pattern(pat: (Pat, Pat)) {
    match pat {
        (Pat::A(Some(_)), Pat::B(_)) => say("matched A::Some + B"),
        (Pat::B(_), Pat::C(_)) => say("matched B and C"),
        _ => say("matched others"),
    }
}

fn joint_pattern_with_or(pat: (Pat, Pat)) {
    match pat {
        (Pat::A(Some(_)) | Pat::C(_), Pat::B(_)) => say("matched A::Some | C + B"),
        (Pat::B(_), Pat::C(_)) => say("matched B and C"),
        _ => say("matched others"),
    }
}

fn joint_or_patterns(pat: (Pat, Pat)) {
    match pat {
        (Pat::A(Some(_)) | Pat::C(_), Pat::B(_) | Pat::C(_)) => say("matched A::Some | C + B | C"),
        (Pat::B(_), Pat::C(_)) => say("matched B and C"),
        _ => say("matched others"),
    }

    // Try to use the matched value
    match pat {
        (Pat::A(Some(a)) | Pat::C(a), Pat::B(b) | Pat::C(b)) => {
            say(&format!("matched A::Some | C ({a}) + B | C ({b})"))
        }
        (Pat::B(_), Pat::C(_)) => say("matched B and C"),
        _ => say("matched others"),
    }
}

fn partial_matched(arr: &[i32]) {
    match arr {
        [selected] | [_, selected] => say(&format!("match arm 1: {selected}")),
        [_, _, selected, ..] => say(&format!("match arm 2: {selected}")),
        _ => say("matched others"),
    }
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

    partial_matched(&[1]);
    partial_matched(&[1, 2, 3]);
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}
