//@ revisions: allowed disallowed
//@[allowed] check-pass
//@ edition: 2024

#![feature(if_let_guard)]
#![cfg_attr(allowed, allow(irrefutable_let_patterns))]
#![cfg_attr(disallowed, deny(irrefutable_let_patterns))]

use std::ops::Range;

fn main() {
    let opt = Some(None..Some(1));

    if let first = &opt && let Some(second) = first && let None = second.start {}
    //[disallowed]~^ ERROR leading irrefutable pattern in let chain

    // No lint as the irrefutable pattern is surrounded by other stuff
    if 4 * 2 == 0 && let first = &opt && let Some(second) = first && let None = second.start {}

    if let first = &opt && let (a, b) = (1, 2) {}
    //[disallowed]~^ ERROR irrefutable `if let` patterns

    if let first = &opt && let Some(second) = first && let None = second.start && let v = 0 {}
    //[disallowed]~^ ERROR leading irrefutable pattern in let chain
    //[disallowed]~^^ ERROR trailing irrefutable pattern in let chain

    if let Some(ref first) = opt && let second = first && let _third = second {}
    //[disallowed]~^ ERROR trailing irrefutable patterns in let chain

    if let Range { start: local_start, end: _ } = (None..Some(1)) && let None = local_start {}
    //[disallowed]~^ ERROR leading irrefutable pattern in let chain

    if let (a, b, c) = (Some(1), Some(1), Some(1)) && let None = Some(1) {}
    //[disallowed]~^ ERROR leading irrefutable pattern in let chain

    if let first = &opt && let None = Some(1) {}
    //[disallowed]~^ ERROR leading irrefutable pattern in let chain

    if let Some(ref first) = opt
        && let Range { start: local_start, end: _ } = first
        && let None = local_start {
    }

    match opt {
        Some(ref first) if let second = first && let _third = second && let v = 4 + 4 => {},
        //[disallowed]~^ ERROR irrefutable `if let` guard patterns
        _ => {}
    }

    // No error about leading irrefutable patterns: the expr on the rhs might
    // use the bindings created by the match.
    match opt {
        Some(ref first) if let Range { start: local_start, end: _ } = first
            && let None = local_start => {},
        _ => {}
    }

    match opt {
        Some(ref first) if let Range { start: Some(_), end: local_end } = first
            && let v = local_end && let w = v => {},
        //[disallowed]~^ ERROR trailing irrefutable patterns in let chain
        _ => {}
    }

    // No error, despite the prefix being irrefutable: moving out could change the behaviour,
    // due to possible side effects of the operation.
    while let first = &opt && let Some(second) = first && let None = second.start {}

    while let first = &opt && let (a, b) = (1, 2) {}
    //[disallowed]~^ ERROR irrefutable `while let` patterns

    while let Some(ref first) = opt && let second = first && let _third = second {}
    //[disallowed]~^ ERROR trailing irrefutable patterns in let chain

    while let Some(ref first) = opt
        && let Range { start: local_start, end: _ } = first
        && let None = local_start {
    }

    // No error. An extra nesting level would be required for the `else if`.
    if opt == Some(None..None) {
    } else if let x = opt.clone().map(|_| 1)
        && x == Some(1)
    {}

    if opt == Some(None..None) {
    } else if opt.is_some()
        && let x = &opt
        //[disallowed]~^ ERROR trailing irrefutable pattern in let chain
    {}

    if opt == Some(None..None) {
    } else {
        if let x = opt.clone().map(|_| 1)
        //[disallowed]~^ ERROR leading irrefutable pattern in let chain
            && x == Some(1)
        {}
    }
}
