#![warn(clippy::doc_link_with_quotes)]

fn main() {
    foo()
}

/// Calls ['bar'] uselessly
//~^ ERROR: possible intra-doc link using quotes instead of backticks
//~| NOTE: `-D clippy::doc-link-with-quotes` implied by `-D warnings`
pub fn foo() {
    bar()
}

/// # Examples
/// This demonstrates issue \#8961
/// ```
/// let _ = vec!['w', 'a', 't'];
/// ```
pub fn bar() {}
