// Test that rustfmt will not warn about comments exceeding max width around lifetime.
// See #2526.

// comment comment comment comment comment comment comment comment comment comment comment comment comment
fn foo() -> F<'a> {
    bar()
}
// comment comment comment comment comment comment comment comment comment comment comment comment comment
