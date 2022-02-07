// ignore-emscripten no processes

// revisions: edition_2015 edition_2021
// [edition_2015]edition:2015
// [edition_2021]edition:2021
// [edition_2015]check-fail
// [edition_2021]run-fail
// [edition_2015]error-pattern:there is no argument named `x`
// [edition_2021]error-pattern:internal error: entered unreachable code: x is 5 and y is 0

fn main() {
    let x = 5;
    unreachable!("x is {x} and y is {y}", y = 0);
}
