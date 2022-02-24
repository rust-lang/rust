#![feature(macro_metavar_expr)]

// `curly` = Right hand side curly brackets
// `no_rhs_dollar` = No dollar sign at the right hand side meta variable "function"
// `round` = Left hand side round brackets

macro_rules! curly__no_rhs_dollar__round {
    ( $( $i:ident ),* ) => { ${ count(i) } };
}

macro_rules! curly__no_rhs_dollar__no_round {
    ( $i:ident ) => { ${ count(i) } };
    //~^ ERROR `count` can not be placed inside the inner-most repetition
}

macro_rules! curly__rhs_dollar__round {
    ( $( $i:ident ),* ) => { ${ count($i) } };
    //~^ ERROR could not find an expected `ident` element
    //~| ERROR expected expression, found `$`
}

macro_rules! curly__rhs_dollar__no_round {
    ( $i:ident ) => { ${ count($i) } };
    //~^ ERROR could not find an expected `ident` element
    //~| ERROR expected expression, found `$`
}

macro_rules! no_curly__no_rhs_dollar__round {
    ( $( $i:ident ),* ) => { count(i) };
    //~^ ERROR cannot find function `count` in this scope
    //~| ERROR cannot find value `i` in this scope
}

macro_rules! no_curly__no_rhs_dollar__no_round {
    ( $i:ident ) => { count(i) };
    //~^ ERROR cannot find function `count` in this scope
    //~| ERROR cannot find value `i` in this scope
}

macro_rules! no_curly__rhs_dollar__round {
    ( $( $i:ident ),* ) => { count($i) };
    //~^ ERROR variable 'i' is still repeating at this depth
}

macro_rules! no_curly__rhs_dollar__no_round {
    ( $i:ident ) => { count($i) };
    //~^ ERROR cannot find function `count` in this scope
}

fn main() {
    curly__no_rhs_dollar__round!(a, b, c);
    curly__no_rhs_dollar__no_round!(a);
    curly__rhs_dollar__round!(a, b, c);
    curly__rhs_dollar__no_round!(a);
    no_curly__no_rhs_dollar__round!(a, b, c);
    no_curly__no_rhs_dollar__no_round!(a);
    no_curly__rhs_dollar__round!(a, b, c);
    no_curly__rhs_dollar__no_round!(a);
    //~^ ERROR cannot find value `a` in this scope
}
