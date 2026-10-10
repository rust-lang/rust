// Regression test for the redundant `note: add `#[derive(Debug)]` to `X` or manually
// `impl Debug for X`` that was emitted alongside the `consider annotating X with
// `#[derive(Debug)]`` suggestion. When the derive suggestion is shown, the note is
// redundant and should be suppressed.
//
// See https://github.com/rust-lang/rust/issues/157118

#[derive(Debug)]
struct S<T>(T);

struct X;

fn main() {
    println!("{:?}", S(X)); //~ ERROR `X` doesn't implement `Debug`
}
