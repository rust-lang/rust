pub extern fn f() -> i32 { 1 }
pub extern fn g() -> i32 { 2 }

pub fn get_f() -> extern fn() -> i32 { f }
pub fn get_g() -> extern fn() -> i32 { g }
