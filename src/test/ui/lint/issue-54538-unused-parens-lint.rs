// build-pass (FIXME(62277): could be check-pass?)

#![allow(ellipsis_inclusive_range_patterns)]
#![allow(unreachable_patterns)]
#![allow(unused_variables)]
#![warn(unused_parens)]

fn main() {
    match 1 {
        (_) => {}         //~ WARNING: unnecessary parentheses around pattern
        (y) => {}         //~ WARNING: unnecessary parentheses around pattern
        (ref r) => {}     //~ WARNING: unnecessary parentheses around pattern
        (e @ 1...2) => {} //~ WARNING: unnecessary parentheses around outer pattern
        (1...2) => {}     // Non ambiguous range pattern should not warn
        e @ (3...4) => {} // Non ambiguous range pattern should not warn
    }

    match &1 {
        (e @ &(1...2)) => {} //~ WARNING: unnecessary parentheses around outer pattern
        &(_) => {}           //~ WARNING: unnecessary parentheses around pattern
        e @ &(1...2) => {}   // Ambiguous range pattern should not warn
        &(1...2) => {}       // Ambiguous range pattern should not warn
    }

    match &1 {
        e @ &(1...2) | e @ &(3...4) => {} // Complex ambiguous pattern should not warn
        &_ => {}
    }

    match 1 {
        (_) => {}         //~ WARNING: unnecessary parentheses around pattern
        (y) => {}         //~ WARNING: unnecessary parentheses around pattern
        (ref r) => {}     //~ WARNING: unnecessary parentheses around pattern
        (e @ 1..=2) => {} //~ WARNING: unnecessary parentheses around outer pattern
        (1..=2) => {}     // Non ambiguous range pattern should not warn
        e @ (3..=4) => {} // Non ambiguous range pattern should not warn
    }

    match &1 {
        (e @ &(1..=2)) => {} //~ WARNING: unnecessary parentheses around outer pattern
        &(_) => {}           //~ WARNING: unnecessary parentheses around pattern
        e @ &(1..=2) => {}   // Ambiguous range pattern should not warn
        &(1..=2) => {}       // Ambiguous range pattern should not warn
    }

    match &1 {
        e @ &(1..=2) | e @ &(3..=4) => {} // Complex ambiguous pattern should not warn
        &_ => {}
    }
}
