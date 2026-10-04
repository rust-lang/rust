//@ edition:2021

// Regression test for #123901. We previously ICE'd as we silently
// swallowed an in the `ExprUseVisitor`.

pub fn test(test: &u64, temp: &u64) {
    async |check, a, b| {
        //~^ ERROR type annotations needed
        temp.abs_diff(12);
    };
}

fn main() {}
