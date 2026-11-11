// Regression test for the semantic changes in
// <https://github.com/rust-lang/rust/pull/139042>.

fn main() {
    let x;
    let (0 | _) = x;
    //~^ ERROR used binding `x` isn't initialized
}
