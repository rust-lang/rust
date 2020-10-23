// aux-build:option_helpers.rs

#![warn(clippy::map_unwrap_or)]

#[macro_use]
extern crate option_helpers;

use std::collections::HashMap;

#[rustfmt::skip]
fn option_methods() {
    let opt = Some(1);

    // Check for `option.map(_).unwrap_or(_)` use.
    // Multi-line cases.
    let _ = opt.map(|x| {
        x + 1
    }
    ).unwrap_or(0);
    let _ = opt.map(|x| x + 1)
        .unwrap_or({
            0
        });
    // Single line `map(f).unwrap_or(None)` case.
    let _ = opt.map(|x| Some(x + 1)).unwrap_or(None);
    // Multi-line `map(f).unwrap_or(None)` cases.
    let _ = opt.map(|x| {
        Some(x + 1)
    }
    ).unwrap_or(None);
    let _ = opt
        .map(|x| Some(x + 1))
        .unwrap_or(None);
    // macro case
    let _ = opt_map!(opt, |x| x + 1).unwrap_or(0); // should not lint

    // Should not lint if not copyable
    let id: String = "identifier".to_string();
    let _ = Some("prefix").map(|p| format!("{}.{}", p, id)).unwrap_or(id);
    // ...but DO lint if the `unwrap_or` argument is not used in the `map`
    let id: String = "identifier".to_string();
    let _ = Some("prefix").map(|p| format!("{}.", p)).unwrap_or(id);

    // Check for `option.map(_).unwrap_or_else(_)` use.
    // Multi-line cases.
    let _ = opt.map(|x| {
        x + 1
    }
    ).unwrap_or_else(|| 0);
    let _ = opt.map(|x| x + 1)
        .unwrap_or_else(||
            0
        );
}

fn main() {
    option_methods();
}
