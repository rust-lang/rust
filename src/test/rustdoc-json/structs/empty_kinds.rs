// @is "$.index[*][?(@.name == 'Unit')].inner.kind" '"unit"'
pub struct Unit;
// @is "$.index[*][?(@.name == 'Tuple')].inner.kind" '"tuple"'
pub struct Tuple();
// @is "$.index[*][?(@.name == 'NamedFields')].inner.kind" '"named_fields"'
pub struct NamedFields {}
