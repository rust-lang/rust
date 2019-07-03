// build-pass (FIXME(62277): could be check-pass?)

/// ```
/// \__________pkt->size___________/          \_result->size_/ \__pkt->size__/
/// ```
pub fn foo() {}

/// ```
///    |
/// LL | use foobar::Baz;
///    |     ^^^^^^ did you mean `baz::foobar`?
/// ```
pub fn bar() {}

/// ```
/// valid
/// ```
///
/// ```
/// \_
/// ```
///
/// ```text
/// "invalid
/// ```
pub fn valid_and_invalid() {}

/// This is a normal doc comment, but...
///
/// There's a code block with bad syntax in it:
///
/// ```rust
/// \_
/// ```
///
/// Good thing we tested it!
pub fn baz() {}

/// Indented block start
///
///     code with bad syntax
///     \_
///
/// Indented block end
pub fn quux() {}

/// Unclosed fence
///
/// ```
/// slkdjf
pub fn xyzzy() {}

/// Indented code that contains a fence
///
///     ```
pub fn blah() {}

/// ```edition2018
/// \_
/// ```
pub fn blargh() {}

#[doc = "```"]
/// \_
#[doc = "```"]
pub fn crazy_attrs() {}
