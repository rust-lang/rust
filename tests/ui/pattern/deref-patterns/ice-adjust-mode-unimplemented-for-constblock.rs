// compile-flags: --error-format=json
// error-pattern: expected a pattern
// or
//~ ERROR expected a pattern

//~ ERROR expected a pattern, found a function call
//~ ERROR usage of qualified paths in this context is experimental
//~ WARN the feature `deref_patterns` is incomplete
//~ ERROR expected tuple struct or tuple variant, found associated function

#![feature(deref_patterns)]

fn main() {
    let vec![const { vec![] }]: Vec<usize> = vec![];
}
