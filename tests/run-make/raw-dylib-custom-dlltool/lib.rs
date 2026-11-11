#[link(name = "extern_1", kind = "raw-dylib")]
extern "C" {
    fn extern_fn_1();
}

pub fn library_function() {
    unsafe {
        extern_fn_1();
    }
}
