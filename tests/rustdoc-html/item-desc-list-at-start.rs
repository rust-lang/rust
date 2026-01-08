//@ has item_desc_list_at_start/index.html
//@ count - '//dl[@class="item-table"]/dd//ul' 0
//@ count - '//dl[@class="item-table"]/dd//li' 0
//@ count - '//dl[@class="item-table"]/dd' 1
//@ snapshot item-table - '//dl[@class="item-table"]'

// based on https://docs.rs/gl_constants/0.1.1/src/gl_constants/lib.rs.html#16

/// * Groups: `SamplePatternSGIS`, `SamplePatternEXT`
pub const MY_CONSTANT: usize = 0;
