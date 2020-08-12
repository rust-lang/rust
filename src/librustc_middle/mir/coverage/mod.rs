//! Metadata from source code coverage analysis and instrumentation.

/// Positional arguments to `libcore::count_code_region()`
pub mod count_code_region_args {
    pub const FUNCTION_SOURCE_HASH: usize = 0;
    pub const COUNTER_ID: usize = 1;
    pub const FILE_NAME: usize = 2;
    pub const START_LINE: usize = 3;
    pub const START_COL: usize = 4;
    pub const END_LINE: usize = 5;
    pub const END_COL: usize = 6;
}

/// Positional arguments to `libcore::coverage_counter_add()` and
/// `libcore::coverage_counter_subtract()`
pub mod coverage_counter_expression_args {
    pub const EXPRESSION_ID: usize = 0;
    pub const LEFT_ID: usize = 1;
    pub const RIGHT_ID: usize = 2;
    pub const FILE_NAME: usize = 3;
    pub const START_LINE: usize = 4;
    pub const START_COL: usize = 5;
    pub const END_LINE: usize = 6;
    pub const END_COL: usize = 7;
}

/// Positional arguments to `libcore::coverage_unreachable()`
pub mod coverage_unreachable_args {
    pub const FILE_NAME: usize = 0;
    pub const START_LINE: usize = 1;
    pub const START_COL: usize = 2;
    pub const END_LINE: usize = 3;
    pub const END_COL: usize = 4;
}
