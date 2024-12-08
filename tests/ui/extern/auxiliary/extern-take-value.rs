pub extern "C" fn f() -> i32 { 1 }
pub extern "C" fn g() -> i32 { 2 }

pub fn get_f() -> extern "C" fn() -> i32 { f }
pub fn get_g() -> extern "C" fn() -> i32 { g }
