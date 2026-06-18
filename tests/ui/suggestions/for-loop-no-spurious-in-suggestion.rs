// Regression test for https://github.com/rust-lang/rust/issues/103561: when the
// `for` loop header already contains an `in`, a missing `in` is not the problem,
// so we must not suggest inserting another one (which would not compile).

fn main() {
    for i i in 0..10 {}
    //~^ ERROR missing `in` in `for` loop
    //~| ERROR expected `{`, found keyword `in`
}
