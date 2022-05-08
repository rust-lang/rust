// Beginners often write an incorrect type name whose initial letter is
// a lowercase while the correct one is an uppercase.
// (e.g. `string` instead of `String`)
// This tests that we suggest the latter when we encounter the former.

fn main() {
    let _ = string::new();
    //~^ ERROR failed to resolve: use of undeclared crate or module `string`
}
