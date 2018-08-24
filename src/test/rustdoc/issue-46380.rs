// compile-flags: --document-private-items

// @has issue_46380/struct.Hidden.html
#[doc(hidden)]
pub struct Hidden;
