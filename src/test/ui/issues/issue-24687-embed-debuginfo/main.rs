// run-pass
// aux-build:issue-24687-lib.rs
// compile-flags:-g

extern crate issue_24687_lib as d;

fn main() {
    // Create a `D`, which has a destructor whose body will be codegen'ed
    // into the generated code here, and thus the local debuginfo will
    // need references into the original source locations from
    // `importer` above.
    let _d = d::D("Hi");
}
