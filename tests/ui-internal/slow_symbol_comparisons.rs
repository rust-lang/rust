#![feature(rustc_private)]
#![warn(clippy::slow_symbol_comparisons)]

extern crate rustc_span;

use clippy_utils::sym;
use rustc_span::Symbol;

fn main() {
    let symbol = sym!(example);
    let other_symbol = sym!(other_example);

    // Should lint
    let slow_comparison = symbol == Symbol::intern("example");
    //~^ error: comparing `Symbol` via `Symbol::intern`
    let slow_comparison_macro = symbol == sym!(example);
    //~^ error: comparing `Symbol` via `Symbol::intern`
    let slow_comparison_backwards = sym!(example) == symbol;
    //~^ error: comparing `Symbol` via `Symbol::intern`

    // Should not lint
    let faster_comparison = symbol.as_str() == "other_example";
    let preinterned_comparison = symbol == other_symbol;
}
