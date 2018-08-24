// @has must_use/struct.Struct.html //pre '#[must_use]'
#[must_use]
pub struct Struct {
    field: i32,
}

// @has must_use/enum.Enum.html //pre '#[must_use = "message"]'
#[must_use = "message"]
pub enum Enum {
    Variant(i32),
}
