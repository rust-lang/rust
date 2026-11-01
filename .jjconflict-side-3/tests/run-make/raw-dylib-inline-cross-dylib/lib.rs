#[link(name = "extern_1", kind = "raw-dylib")]
extern "C" {
    fn extern_fn_1();
    fn extern_fn_2();
}

#[inline]
pub fn inline_library_function() {
    unsafe {
        extern_fn_1();
        extern_fn_2();
    }
}

pub fn library_function() {
    unsafe {
        extern_fn_2();
    }
}
