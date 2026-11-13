//@ run-rustfix
//@ edition:2018

// Regression test for issue 79694

fn main() {
    let _ = move async { }; //~ ERROR the order of `move` and `async` is incorrect
}
