// compile-flags: -Zunstable-options --document-hidden-items

// @has issue_67851_hidden/struct.Hidden.html
#[doc(hidden)]
pub struct Hidden;

// @!has issue_67851_hidden/struct.Private.html
struct Private;
