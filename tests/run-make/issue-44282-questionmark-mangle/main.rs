#![crate_type = "cdylib"]

#[no_mangle]
pub extern "C" fn foo(a: i32, b: i32) -> i32 {
    1
}

#[export_name = "?bar@@YAXXZ"]
pub extern "C" fn bar(a: i32, b: i32) -> i32 {
    2
}
