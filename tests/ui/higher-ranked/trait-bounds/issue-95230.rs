// This used to be a test for overflow handling + higher-ranked outlives
// in the new solver, but this test isn't expected to pass since WF preds
// are not coinductive anymore.

pub struct Bar
where
    for<'a> &'a mut Self:;
//~^ ERROR overflow evaluating the requirement `for<'a> &'a mut Bar well-formed`

fn main() {}
