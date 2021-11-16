// run-rustfix
// edition:2018

// Regression test for issue 79694

fn main() {
    let _ = async move { }; //~ ERROR 7:13: 7:23: the order of `move` and `async` is incorrect
}
