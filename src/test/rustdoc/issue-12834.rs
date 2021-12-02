// Tests that failing to syntax highlight a rust code-block doesn't cause
// rustdoc to fail, while still rendering the code-block (without highlighting).

#![allow(rustdoc::invalid_rust_codeblocks)]

// @has issue_12834/fn.foo.html
// @has - //pre 'a + b '

/// ```
/// a + b ∈ Self ∀ a, b ∈ Self
/// ```
pub fn foo() {}
