// General syntax errors that apply to all matavariable expressions
//
// We don't invoke the macros here to ensure code gets rejected at the definition rather than
// only when expanded.

#![feature(macro_metavar_expr)]

macro_rules! dollar_dollar_in_the_lhs {
    ( $$ $a:ident ) => {
        //~^ ERROR unexpected token: $
    };
}

macro_rules! metavar_in_the_lhs {
    ( ${ len() } ) => {
        //~^ ERROR unexpected token: {
        //~| ERROR expected one of: `*`, `+`, or `?`
    };
}

macro_rules! metavar_token_without_ident {
    ( $( $i:ident ),* ) => { ${ ignore() } };
    //~^ ERROR meta-variable expressions must be referenced using a dollar sign
}

macro_rules! metavar_with_literal_suffix {
    ( $( $i:ident ),* ) => { ${ index(1u32) } };
    //~^ ERROR only unsuffixes integer literals are supported in meta-variable expressions
}

macro_rules! mve_without_parens {
    ( $( $i:ident ),* ) => { ${ count } };
    //~^ ERROR expected `(`
}

#[rustfmt::skip]
macro_rules! empty_expression {
    () => { ${} };
    //~^ ERROR expected identifier or string literal
}

#[rustfmt::skip]
macro_rules! open_brackets_with_lit {
     () => { ${ "hi" } };
     //~^ ERROR expected identifier
 }

macro_rules! mvs_missing_paren {
    ( $( $i:ident ),* ) => { ${ count $i ($i) } };
    //~^ ERROR expected `(`
}

macro_rules! mve_wrong_delim {
    ( $( $i:ident ),* ) => { ${ count{i} } };
    //~^ ERROR expected `(`
}

macro_rules! invalid_metavar {
    () => { ${ignore($123)} }
    //~^ ERROR expected identifier, found `123`
}

#[rustfmt::skip]
macro_rules! open_brackets_with_group {
    ( $( $i:ident ),* ) => { ${ {} } };
    //~^ ERROR expected identifier
}

macro_rules! extra_garbage_after_metavar {
    ( $( $i:ident ),* ) => {
        ${count() a b c}
        //~^ ERROR unexpected trailing tokens
        ${count($i a b c)}
        //~^ ERROR unexpected trailing tokens
        ${count($i, 1 a b c)}
        //~^ ERROR unexpected trailing tokens
        ${count($i) a b c}
        //~^ ERROR unexpected trailing tokens

        ${ignore($i) a b c}
        //~^ ERROR unexpected trailing tokens
        ${ignore($i a b c)}
        //~^ ERROR unexpected trailing tokens

        ${index() a b c}
        //~^ ERROR unexpected trailing tokens
        ${index(1 a b c)}
        //~^ ERROR unexpected trailing tokens

        ${index() a b c}
        //~^ ERROR unexpected trailing tokens
        ${index(1 a b c)}
        //~^ ERROR unexpected trailing tokens
        ${index(1, a b c)}
        //~^ ERROR unexpected trailing tokens
    };
}

const IDX: usize = 1;
macro_rules! metavar_depth_is_not_literal {
    ( $( $i:ident ),* ) => { ${ index(IDX) } };
    //~^ ERROR meta-variable expression depth must be a literal
}

macro_rules! unknown_count_ident {
    ( $( $i:ident )* ) => {
        ${count(foo)}
        //~^ ERROR meta-variable expressions must be referenced using a dollar sign
    };
}

macro_rules! unknown_ignore_ident {
    ( $( $i:ident )* ) => {
        ${ignore(bar)}
        //~^ ERROR meta-variable expressions must be referenced using a dollar sign
    };
}

macro_rules! unknown_metavar {
    ( $( $i:ident ),* ) => { ${ aaaaaaaaaaaaaa(i) } };
    //~^ ERROR unrecognized metavariable expression
}

fn main() {}
