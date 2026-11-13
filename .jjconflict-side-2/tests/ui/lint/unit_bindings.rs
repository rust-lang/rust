//! Basic checks for `unit_bindings` lint.
//!
//! The `unit_bindings` lint tries to detect cases like `let list = list.sort()`. The lint will
//! trigger on bindings that have the unit `()` type **except** if:
//!
//! - The user wrote `()` on either side, i.e.
//!     - `let () = <expr>;` or `let <expr> = ();`
//!     - `let _ = ();`
//! - The binding occurs within macro expansions, e.g. `foo!();`.
//! - The user explicitly provided type annotations, e.g. `let x: () = <expr>`.
//!
//! Examples where the lint *should* fire on include:
//!
//! - `let _ = <expr>;`
//! - `let pat = <expr>;`
//! - `let _pat = <expr>;`

//@ revisions: default_level deny_level
//@[default_level] check-pass (`unit_bindings` is currently allow-by-default)

#![allow(unused)]
#![cfg_attr(deny_level, deny(unit_bindings))]

// The `list` binding below should trigger the lint if it's not contained in a macro expansion.
macro_rules! expands_to_sus {
    () => {
        let mut v = [1, 2, 3];
        let list = v.sort();
    }
}

// No warning for `y` and `z` because it is provided as type parameter.
fn ty_param_check<T: Copy>(x: T) {
    let y = x;
    let z: T = x;
}

fn main() {
    // No warning if user explicitly wrote `()` on either side.
    let expr = ();
    let () = expr;
    let _ = ();
    // No warning if user explicitly annotates the unit type on the binding.
    let pat: () = expr;
    // No warning for let bindings with unit type in macro expansions.
    expands_to_sus!();
    // No warning for unit bindings in generic fns.
    ty_param_check(());

    let _ = expr; //[deny_level]~ ERROR binding has unit type
    let pat = expr; //[deny_level]~ ERROR binding has unit type
    let _pat = expr; //[deny_level]~ ERROR binding has unit type

    let mut v = [1, 2, 3];
    let list = v.sort(); //[deny_level]~ ERROR binding has unit type

    // Limitation: the lint currently does not fire on nested unit LHS bindings, i.e.
    // this will not currently trigger the lint.
    let (nested, _) = (expr, 0i32);
}
