/// I want...
///
/// # Anchor!
pub struct Something;

// @has intra_links_anchors/struct.SomeOtherType.html
// @has - '//a/@href' '../intra_links_anchors/struct.Something.html#Anchor!'

/// I want...
///
/// To link to [Something#Anchor!]
pub struct SomeOtherType;
