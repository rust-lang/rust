//@ revisions: edition2021 edition2024
//@ [edition2021] edition: 2021
//@ [edition2024] edition: 2024

fn main() {
    if let true = true || false {}
    //~^ ERROR `||` operators are not supported in let chain conditions
    // With parentheses
    if (let true = true) || false {}
    //~^ ERROR expected expression, found `let` statement
    // Multiple || operators
    if let true = true || false || true {}
    //~^ ERROR `||` operators are not supported in let chain conditions
    // Mixed operators (should still show error for ||)
    if let true = true && false || true {}
    //~^ ERROR `||` operators are not supported in let chain conditions
}
