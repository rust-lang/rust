// Issue 46112: An extern crate pub re-exporting libcore was causing
// paths rooted from `std` to be misrendered in the diagnostic output.

// ignore-windows
// aux-build:xcrate-issue-43189-a.rs
// aux-build:xcrate-issue-43189-b.rs

extern crate xcrate_issue_43189_b;
fn main() {
    ().a();
    //~^ ERROR no method named `a` found
}
