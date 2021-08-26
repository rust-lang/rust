/// I want...
///
/// # Anchor!
pub struct Something;

// @has anchors/struct.SomeOtherType.html
// @has - '//a/@href' 'struct.Something.html#Anchor!'

/// I want...
///
/// To link to [Something#Anchor!]
pub struct SomeOtherType;

/// Primitives?
///
/// [u32#hello]
// @has anchors/fn.x.html
// @has - '//a/@href' '{{channel}}/std/primitive.u32.html#hello'
pub fn x() {}

/// [prim@usize#x]
// @has anchors/usize/index.html
// @has - '//a/@href' '{{channel}}/std/primitive.usize.html#x'
pub mod usize {}
