// compile-flags: -Zunstable-options --document-private-items --document-hidden-items

// @has issue_67851_both/struct.Hidden.html
#[doc(hidden)]
pub struct Hidden;

// @has issue_67851_both/struct.Private.html
struct Private;
