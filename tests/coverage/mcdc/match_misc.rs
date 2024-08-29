#![feature(coverage_attribute)]
//@ edition: 2021
//@ min-llvm-version: 19
//@ compile-flags: -Zcoverage-options=mcdc
//@ llvm-cov-flags: --show-branches=count --show-mcdc

// Loop over iterator contains pattern matching implicitly,
// do not generate mappings for it.
fn loop_over_iterator() {
    for val in [1, 2, 3] {
        say(&val.to_string());
    }
}

// Macro makes all conditions share same span.
// But now we don't generate mappings for it
fn match_with_macros(val: i32) {
    macro_rules! variant_identifier {
    (
     $val:expr,  ($($index:expr),*)
    )=> {
        match $val {
            $(
                $index => say(&format!("{}",$index)),
            )*
            _ => say("not matched"),
        }
    }
}
    variant_identifier!(val, (0, 1, 2));
}

// No match pairs when lowering matching tree.
fn empty_matching_decision(val: i32) {
    match val {
        x if x > 8 && x < 10 => say("in (8, 10)"),
        x if x > 4 && x < 7 => say("in (4, 7)"),
        _ => say("other"),
    }
}

// Matching decision skips the first candidate
fn skipped_matching_decision(val: i32) {
    match val {
        x if x >= 0 => say("non-negative"),
        -1 => say("-1"),
        _ => say("other"),
    }
}

// The first two candidates share same condition.
fn overlapping_decisions(val: (Option<i32>, Option<i32>)) {
    match val {
        (Some(_), Some(_)) => say("both some"),
        (Some(_), None) | (None, Some(_)) => say("one and only one some"),
        (None, None) => say("none"),
    }
}

fn partial_matched_decision(val: u8) {
    // `b'-'` is the second test while `b'0'..=b'9'` is the last, though they
    // are in same candidate.
    match val {
        b'"' | b'r' => say("quote or r"),
        b'0'..=b'9' | b'-' => say("number or -"),
        b't' | b'f' => say("t or f"),
        _ => {}
    }
}

// Patterns are tested with several basic blocks.
fn partial_matched_with_several_blocks(val: u8) {
    match val {
        b'a'..=b'f' => say("hex"),
        b'A'..=b'F' => say("hex upper"),
        b'_' => say("underscore"),
        _ => say("break"),
    }
}

fn match_failure_test_kind(val: bool, opt: Option<i32>) {
    match (val, opt) {
        (false, None) => say("none"),
        (false, Some(_)) => say("some"),
        _ => say("other"),
    }
}

enum Pat {
    A(i32),
    B(i32),
}

// The last arm is shown like a condition but it never fails if tested.
fn uncoverable_condition(val: (Pat, Pat)) {
    match val {
        (Pat::A(a), Pat::A(b)) => say(&(a + b).to_string()),
        (Pat::B(a), _) | (_, Pat::B(a)) => say(&a.to_string()),
    }
}

fn nested_matching(a: bool, val: Pat) {
    if a && match val {
        Pat::A(x) => x == 2,
        _ => false,
    } {
        say("yes");
    }
}

// It's possible to match two arms once.
fn multi_matched_candidates(val: Pat, a: i32) {
    match val {
        Pat::A(f) if f == a => say("first"),
        Pat::A(1) if a > 0 => say("second"),
        _ => say("other"),
    }
}

fn empty_subcandidate(val: Pat) {
    match val {
        Pat::A(1) | Pat::A(2) => say("first"),
        // The first two condition in this pattern is redundant indeed.
        // But this piece of code is legitimate and it could cause a subcandidate
        // with no match pair.
        Pat::A(_) | Pat::B(_) | _ => say("other"),
    }
}

fn implicit_folded_condition(val: (bool, bool)) {
    match val {
        // The first `true` is always matched if tested.
        (false, false) | (true, true) => say("same"),
        _ => say("not same"),
    }
}

#[coverage(off)]
fn main() {
    loop_over_iterator();

    match_with_macros(0);
    match_with_macros(2);
    match_with_macros(5);

    empty_matching_decision(12);
    empty_matching_decision(5);

    skipped_matching_decision(-1);
    skipped_matching_decision(-5);

    overlapping_decisions((Some(1), Some(2)));
    overlapping_decisions((Some(1), None));
    overlapping_decisions((None, None));

    partial_matched_decision(b'"');
    partial_matched_decision(b'r');
    partial_matched_decision(b'7');
    partial_matched_decision(b'-');

    partial_matched_with_several_blocks(b'd');
    partial_matched_with_several_blocks(b'D');
    partial_matched_with_several_blocks(b'_');

    match_failure_test_kind(false, None);
    match_failure_test_kind(false, Some(1));

    uncoverable_condition((Pat::A(1), Pat::A(2)));
    uncoverable_condition((Pat::B(1), Pat::B(2)));

    nested_matching(true, Pat::A(1));
    nested_matching(true, Pat::A(2));
    nested_matching(false, Pat::A(2));

    multi_matched_candidates(Pat::A(1), 1);
    multi_matched_candidates(Pat::A(1), 8);

    empty_subcandidate(Pat::A(1));
    empty_subcandidate(Pat::B(1));

    implicit_folded_condition((false, false));
    implicit_folded_condition((false, true));
    implicit_folded_condition((true, false));
    implicit_folded_condition((true, true));
}

#[coverage(off)]
fn say(message: &str) {
    core::hint::black_box(message);
}
