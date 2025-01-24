// This check veifies that we do not ICE when not showing a user type
// in the suggestions/diagnostics.
//
// cf. https://github.com/rust-lang/rust/issues/134345
//
//@ check-pass

struct A;

fn fna(_a: A) {}

#[allow(unpredictable_function_pointer_comparisons)]
fn main() {
    let fa: fn(A) = fna;
    let _ = fa == fna;
}
