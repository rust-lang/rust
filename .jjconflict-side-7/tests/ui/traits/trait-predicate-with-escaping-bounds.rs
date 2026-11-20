//! Regression test for <https://github.com/rust-lang/rust/issues/163206>.
//@ check-fail

struct StatSeries {}

impl StatSeries {
    fn new<H>(headers: [H; N]) -> () //~ ERROR: cannot find value `N` in this scope [E0425]
    where
        String: for<'a> From<&'a H>,
    {
    }
}

fn main() {
    let series = StatSeries::new([""]); //~ ERROR: the trait bound `for<'a> String: From<&'a &str>` is not satisfied [E0277]
}
