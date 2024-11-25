//@ revisions: rpass1 rpass2
//@ aux-build:invalid-span-helper-lib.rs

// This issue has several different parts. The high level idea is:
// 1. We create an 'invalid' span with the help of the `respan` proc-macro,
// The compiler attempts to prevent the creation of invalid spans by
// refusing to join spans with different `SyntaxContext`s. We work around
// this by applying the same `SyntaxContext` to the span of every token,
// using `Span::resolved_at`
// 2. We using this invalid span in the body of a function, causing it to get
// encoded into the `optimized_mir`
// 3. We call the function from a different crate - since the function is generic,
// monomorphization runs, causing `optimized_mir` to get called.
// 4. We re-run compilation using our populated incremental cache, but without
// making any changes. When we recompile the crate containing our generic function
// (`invalid_span_helper_lib`), we load the span from the incremental cache, and
// write it into the crate metadata.

extern crate invalid_span_helper_lib;

fn main() {
    invalid_span_helper_lib::foo::<u8>();
}
