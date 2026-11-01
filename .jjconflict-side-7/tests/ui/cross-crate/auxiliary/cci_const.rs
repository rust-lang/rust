pub extern "C" fn bar() {
}

pub const foopy: &'static str = "hi there";
pub const uint_val: usize = 12;
pub const uint_expr: usize = (1 << uint_val) - 1;
