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
    //~^ ERROR expected identifier, found `$`
    //~| ERROR expected expression, found `$`
}

macro_rules! curly__rhs_dollar__no_round {
    ( $i:ident ) => { ${ count($i) } };
    //~^ ERROR expected identifier, found `$`
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

// Other scenarios

macro_rules! dollar_dollar_in_the_lhs {
    ( $$ $a:ident ) => {
    //~^ ERROR unexpected token: $
    };
}

macro_rules! extra_garbage_after_metavar {
    ( $( $i:ident ),* ) => {
        ${count() a b c}
        //~^ ERROR unexpected token: a
        //~| ERROR expected expression, found `$`
        ${count(i a b c)}
        //~^ ERROR unexpected token: a
        ${count(i, 1 a b c)}
        //~^ ERROR unexpected token: a
        ${count(i) a b c}
        //~^ ERROR unexpected token: a

        ${ignore(i) a b c}
        //~^ ERROR unexpected token: a
        ${ignore(i a b c)}
        //~^ ERROR unexpected token: a

        ${index() a b c}
        //~^ ERROR unexpected token: a
        ${index(1 a b c)}
        //~^ ERROR unexpected token: a

        ${index() a b c}
        //~^ ERROR unexpected token: a
        ${index(1 a b c)}
        //~^ ERROR unexpected token: a
    };
}

const IDX: usize = 1;
macro_rules! metavar_depth_is_not_literal {
    ( $( $i:ident ),* ) => { ${ index(IDX) } };
    //~^ ERROR meta-variable expression depth must be a literal
    //~| ERROR expected expression, found `$`
}

macro_rules! metavar_in_the_lhs {
    ( ${ length() } ) => {
        //~^ ERROR unexpected token: {
        //~| ERROR expected one of: `*`, `+`, or `?`
    };
}

macro_rules! metavar_token_without_ident {
    ( $( $i:ident ),* ) => { ${ ignore() } };
    //~^ ERROR expected identifier
    //~| ERROR expected expression, found `$`
}

macro_rules! metavar_with_literal_suffix {
    ( $( $i:ident ),* ) => { ${ index(1u32) } };
    //~^ ERROR only unsuffixes integer literals are supported in meta-variable expressions
    //~| ERROR expected expression, found `$`
}

macro_rules! metavar_without_parens {
    ( $( $i:ident ),* ) => { ${ count{i} } };
    //~^ ERROR meta-variable expression parameter must be wrapped in parentheses
    //~| ERROR expected expression, found `$`
}

macro_rules! open_brackets_without_tokens {
    ( $( $i:ident ),* ) => { ${ {} } };
    //~^ ERROR expected expression, found `$`
    //~| ERROR expected identifier
}

macro_rules! unknown_count_ident {
    ( $( $i:ident )* ) => {
        ${count(foo)}
        //~^ ERROR variable `foo` is not recognized in meta-variable expression
    };
}

macro_rules! unknown_ignore_ident {
    ( $( $i:ident )* ) => {
        ${ignore(bar)}
        //~^ ERROR variable `bar` is not recognized in meta-variable expression
    };
}

macro_rules! unknown_metavar {
    ( $( $i:ident ),* ) => { ${ aaaaaaaaaaaaaa(i) } };
    //~^ ERROR unrecognized meta-variable expression
    //~| ERROR expected expression
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

    extra_garbage_after_metavar!(a);
    metavar_depth_is_not_literal!(a);
    metavar_token_without_ident!(a);
    metavar_with_literal_suffix!(a);
    metavar_without_parens!(a);
    open_brackets_without_tokens!(a);
    unknown_count_ident!(a);
    unknown_ignore_ident!(a);
    unknown_metavar!(a);
}
