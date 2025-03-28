//@ needs-subprocess

//@ revisions: edition_2015 edition_2021
//@ [edition_2015]edition:2015
//@ [edition_2021]edition:2021
//@ [edition_2015]run-fail
//@ [edition_2021]check-fail
//@ [edition_2015]error-pattern:internal error: entered unreachable code: hello

#![allow(non_fmt_panics)]

fn main() {
    let a = "hello";
    unreachable!(a); //[edition_2021]~ ERROR format argument must be a string literal
}
