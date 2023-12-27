// @has issue_118180_empty_tuple_struct/enum.Enum.html
pub enum Enum {
    // @has - '//*[@id="variant.Empty"]//h3' 'Empty()'
    Empty(),
}

// @has issue_118180_empty_tuple_struct/struct.Empty.html
// @has - '//pre/code' 'Empty()'
pub struct Empty();
