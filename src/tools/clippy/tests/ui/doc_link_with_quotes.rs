#![warn(clippy::doc_link_with_quotes)]

fn main() {
    foo()
}

/// Calls ['bar'] uselessly
pub fn foo() {
    bar()
}

/// # Examples
/// This demonstrates issue \#8961
/// ```
/// let _ = vec!['w', 'a', 't'];
/// ```
pub fn bar() {}
