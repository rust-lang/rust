//@ aux-build:issue-29181.rs

extern crate issue_29181 as foo;

fn main() {
    0.homura(); //~ ERROR no method named `homura` found
    // Issue #47759, detect existing method on the fundamental impl:
    let _ = |x: f64| x * 2.0.exp(); //~ ERROR can't call method `exp` on ambiguous numeric type
}
