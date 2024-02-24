//@ compile-flags: --document-private-items

// @!has issue_67851_private/struct.Hidden.html
#[doc(hidden)]
pub struct Hidden;

// @has issue_67851_private/struct.Private.html
struct Private;
