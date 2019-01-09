// aux-build:option_helpers.rs

//! Checks implementation of `RESULT_MAP_UNWRAP_OR_ELSE`

#![warn(clippy::result_map_unwrap_or_else)]

#[macro_use]
extern crate option_helpers;

fn result_methods() {
    let res: Result<i32, ()> = Ok(1);

    // Check RESULT_MAP_UNWRAP_OR_ELSE
    // single line case
    let _ = res.map(|x| x + 1).unwrap_or_else(|e| 0); // should lint even though this call is on a separate line
                                                      // multi line cases
    let _ = res.map(|x| x + 1).unwrap_or_else(|e| 0);
    let _ = res.map(|x| x + 1).unwrap_or_else(|e| 0);
    // macro case
    let _ = opt_map!(res, |x| x + 1).unwrap_or_else(|e| 0); // should not lint
}

fn main() {}
