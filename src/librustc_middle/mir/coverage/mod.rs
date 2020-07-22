//! Metadata from source code coverage analysis and instrumentation.

/// Positional arguments to `libcore::count_code_region()`
pub mod count_code_region_args {
    pub const COUNTER_INDEX: usize = 0;
    pub const START_BYTE_POS: usize = 1;
    pub const END_BYTE_POS: usize = 2;
}

/// Positional arguments to `libcore::coverage_counter_add()` and
/// `libcore::coverage_counter_subtract()`
pub mod coverage_counter_expression_args {
    pub const COUNTER_EXPRESSION_INDEX: usize = 0;
    pub const LEFT_INDEX: usize = 1;
    pub const RIGHT_INDEX: usize = 2;
    pub const START_BYTE_POS: usize = 3;
    pub const END_BYTE_POS: usize = 4;
}

/// Positional arguments to `libcore::coverage_unreachable()`
pub mod coverage_unreachable_args {
    pub const START_BYTE_POS: usize = 0;
    pub const END_BYTE_POS: usize = 1;
}
