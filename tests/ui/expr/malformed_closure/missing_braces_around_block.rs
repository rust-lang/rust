// This snippet ensures that no attempt to recover on a semicolon instead of
// comma is made next to a closure body.
//
// If this recovery happens, then plenty of errors are emitted. Here, we expect
// only one error.
//
// This is part of the following issues:
// https://github.com/rust-lang/rust/issues/88065
// https://github.com/rust-lang/rust/issues/107959

//@ run-rustfix

fn main() {
    // Closure with multiple expressions delimited by semicolon.
    let num = 5;
    (1..num).reduce(|a, b|
        //~^ ERROR: closure bodies that contain statements must be surrounded by braces
        println!("{}", a);
        a * b
    ).unwrap();

    // Closure with a single expression ended by a semicolon.
    let mut v = vec![1, 2, 3];
    v.iter_mut().for_each(|x|*x = *x+1;);
        //~^ ERROR: closure bodies that contain statements must be surrounded by braces
}
