// ignore-emscripten no processes

// revisions: edition_2015 edition_2021
// [edition_2015]edition:2015
// [edition_2021]edition:2021
// [edition_2015]run-fail
// [edition_2021]check-fail
// [edition_2015]error-pattern:internal error: entered unreachable code: hello
// [edition_2021]error-pattern:format argument must be a string literal

#![allow(non_fmt_panics)]

fn main() {
    let a = "hello";
    unreachable!(a);
}
