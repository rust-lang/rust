#![cfg_attr(target_arch = "x86", feature(raw_dylib))]

#[link(name = "exporter", kind = "raw-dylib")]
extern "stdcall" {
    #[link_ordinal(15)]
    fn imported_function_stdcall(i: i32);
}

#[link(name = "exporter", kind = "raw-dylib")]
extern "fastcall" {
    #[link_ordinal(18)]
    fn imported_function_fastcall(i: i32);
}

pub fn library_function() {
    unsafe {
        imported_function_stdcall(6);
        imported_function_fastcall(125);
    }
}
