// Regression test for <https://github.com/rust-lang/rust/issues/61463>
// A test for the issue where the variable meta is mistakenly treated as a reserved keyword.

fn main() {
    let xyz = meta;
    //~^ ERROR cannot find value `meta` in this scope [E0425]
}
