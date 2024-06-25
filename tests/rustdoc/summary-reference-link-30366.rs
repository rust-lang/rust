//@ has issue_30366/index.html '//a/@href' 'http://www.rust-lang.org/'

// https://github.com/rust-lang/rust/issues/30366
#![crate_name="issue_30366"]

/// Describe it. [Link somewhere][1].
///
/// [1]: http://www.rust-lang.org/
pub fn here_is_a_fn() { }
