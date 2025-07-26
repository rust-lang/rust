// Errors for the `count` and `length` metavariable expressions

#![feature(macro_metavar_expr)]

// `curly` = Right hand side curly brackets
// `no_rhs_dollar` = No dollar sign at the right hand side meta variable "function"
// `round` = Left hand side round brackets

macro_rules! curly__no_rhs_dollar__round {
    ( $( $i:ident ),* ) => { ${ count($i) } };
}
const _: u32 = curly__no_rhs_dollar__round!(a, b, c);

macro_rules! curly__no_rhs_dollar__no_round {
    ( $i:ident ) => { ${ count($i) } };
    //~^ ERROR `count` can not be placed inside the innermost repetition
}
curly__no_rhs_dollar__no_round!(a);

macro_rules! curly__rhs_dollar__no_round {
    ( $i:ident ) => { ${ count($i) } };
    //~^ ERROR `count` can not be placed inside the innermost repetition
}
curly__rhs_dollar__no_round !(a);

#[rustfmt::skip] // autoformatters can break a few of the error traces
macro_rules! no_curly__no_rhs_dollar__round {
    ( $( $i:ident ),* ) => { count(i) };
    //~^ ERROR missing `fn` or `struct` for function or struct definition
}
no_curly__no_rhs_dollar__round !(a, b, c);

#[rustfmt::skip] // autoformatters can break a few of the error traces
macro_rules! no_curly__no_rhs_dollar__no_round {
    ( $i:ident ) => { count(i) };
    //~^ ERROR missing `fn` or `struct` for function or struct definition
}
no_curly__no_rhs_dollar__no_round !(a);

#[rustfmt::skip] // autoformatters can break a few of the error traces
macro_rules! no_curly__rhs_dollar__round {
    ( $( $i:ident ),* ) => { count($i) };
    //~^ ERROR variable `i` is still repeating at this depth
}
no_curly__rhs_dollar__round! (a);

#[rustfmt::skip] // autoformatters can break a few of the error traces
macro_rules! no_curly__rhs_dollar__no_round {
    ( $i:ident ) => { count($i) };
    //~^ ERROR cannot find function `count` in this scope
}
const _: u32 = no_curly__rhs_dollar__no_round! (a);
//~^ ERROR cannot find value `a` in this scope

fn main() {}
