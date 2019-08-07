extern "C" { fn printf(format: *const i8, ...) -> i32; }
extern "C" { fn printf(#[attr] format: *const i8, ...) -> i32; }
