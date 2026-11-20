// Errors for the `count` and `length` metavariable expressions

// These are cases where there are no macro parsing errors (which would silence resolve errors).

#![feature(macro_metavar_expr)]

// `curly` = Right hand side curly brackets
// `no_rhs_dollar` = No dollar sign at the right hand side meta variable "function"
// `round` = Left hand side round brackets

#[rustfmt::skip] // autoformatters can break a few of the error traces
macro_rules! no_curly__rhs_dollar__no_round {
    ( $i:ident ) => { count($i) };
    //~^ ERROR cannot find function `count` in this scope
}
const _: u32 = no_curly__rhs_dollar__no_round! (a);
//~^ ERROR cannot find value `a` in this scope

fn main() {}
