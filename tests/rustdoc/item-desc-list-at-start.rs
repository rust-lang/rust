// @has item_desc_list_at_start/index.html
// @count - '//ul[@class="item-table"]/li/div/li' 0
// @count - '//ul[@class="item-table"]/li' 1
// @snapshot item-table - '//ul[@class="item-table"]'

// based on https://docs.rs/gl_constants/0.1.1/src/gl_constants/lib.rs.html#16

/// * Groups: `SamplePatternSGIS`, `SamplePatternEXT`
pub const MY_CONSTANT: usize = 0;
