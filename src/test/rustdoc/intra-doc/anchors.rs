/// I want...
///
/// # Anchor!
pub struct Something;

// @has anchors/struct.SomeOtherType.html
// @has - '//a/@href' '../anchors/struct.Something.html#Anchor!'

/// I want...
///
/// To link to [Something#Anchor!]
pub struct SomeOtherType;
