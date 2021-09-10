// This snippet ensures that no attempt to recover on a semicolon instead of
// comma is made next to a closure body.
//
// If this recovery happens, then plenty of errors are emitted. Here, we expect
// only one error.
//
// This is part of issue #88065:
// https://github.com/rust-lang/rust/issues/88065

// run-rustfix

fn main() {
    let num = 5;
    (1..num).reduce(|a, b|
        //~^ ERROR: closure bodies that contain statements must be surrounded by braces
        println!("{}", a);
        a * b
    ).unwrap();
}
