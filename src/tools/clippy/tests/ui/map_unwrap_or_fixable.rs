// run-rustfix
// aux-build:option_helpers.rs

#![warn(clippy::map_unwrap_or)]

#[macro_use]
extern crate option_helpers;

use std::collections::HashMap;

#[rustfmt::skip]
fn option_methods() {
    let opt = Some(1);

    // Check for `option.map(_).unwrap_or_else(_)` use.
    // single line case
    let _ = opt.map(|x| x + 1)
        // Should lint even though this call is on a separate line.
        .unwrap_or_else(|| 0);

    // Macro case.
    // Should not lint.
    let _ = opt_map!(opt, |x| x + 1).unwrap_or_else(|| 0);

    // Issue #4144
    {
        let mut frequencies = HashMap::new();
        let word = "foo";

        frequencies
            .get_mut(word)
            .map(|count| {
                *count += 1;
            })
            .unwrap_or_else(|| {
                frequencies.insert(word.to_owned(), 1);
            });
    }
}

#[rustfmt::skip]
fn result_methods() {
    let res: Result<i32, ()> = Ok(1);

    // Check for `result.map(_).unwrap_or_else(_)` use.
    // single line case
    let _ = res.map(|x| x + 1)
        // should lint even though this call is on a separate line
        .unwrap_or_else(|_e| 0);

    // macro case
    let _ = opt_map!(res, |x| x + 1).unwrap_or_else(|_e| 0); // should not lint
}

fn main() {
    option_methods();
    result_methods();
}
