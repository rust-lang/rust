// In expression, `&a..=b` is treated as `(&a)..=(b)` and `box a..=b` is
// `(box a)..=(b)`. In a pattern, however, `&a..=b` means `&(a..=b)`. This may
// lead to confusion.
//
// We are going to disallow `&a..=b` and `box a..=b` in a pattern. However, the
// older ... syntax is still allowed as a stability guarantee.

#![feature(box_patterns)]
#![warn(ellipsis_inclusive_range_patterns)]


pub fn main() {
    match &12 {
        &0...9 => {}
        //~^ WARN `...` range patterns are deprecated
        //~| HELP use `..=` for an inclusive range
        &10..=15 => {}
        //~^ ERROR the range pattern here has ambiguous interpretation
        //~^^ HELP add parentheses to clarify the precedence
        &(16..=20) => {}
        _ => {}
    }

    match Box::new(12) {
        box 0...9 => {}
        //~^ WARN `...` range patterns are deprecated
        //~| HELP use `..=` for an inclusive range
        box 10..=15 => {}
        //~^ ERROR the range pattern here has ambiguous interpretation
        //~^^ HELP add parentheses to clarify the precedence
        box (16..=20) => {}
        _ => {}
    }
}
