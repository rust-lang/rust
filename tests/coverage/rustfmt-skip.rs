//@ edition: 2024

// The presence of `#[rustfmt::skip]` on a function should not cause macros
// within that function to mysteriously not be instrumented.
//
// This test detects problems that can occur when building an expansion tree
// based on `ExpnData::parent` instead of `ExpnData::call_site`, for example.

#[rustfmt::skip]
fn main() {
    // Ensure a gap between the body start and the first statement.
    println!(
        // Keep this on a separate line, to distinguish instrumentation of
        // `println!` from instrumentation of its arguments.
        "hello"
    );
}
