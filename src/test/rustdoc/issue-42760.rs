// @has issue_42760/struct.NonGen.html
// @has - '//h1' 'Example'

/// Item docs.
///
#[doc="Hello there!"]
///
/// # Example
///
/// ```rust
/// // some code here
/// ```
pub struct NonGen;
