extern crate raw_dylib_test;
extern crate raw_dylib_test_wrapper;

#[link(name = "extern_2", kind = "raw-dylib")]
extern "C" {
    fn extern_fn_2();
}

fn main() {
    // NOTE: The inlined call to `extern_fn_2` links against the function in extern_2.dll instead
    // of extern_1.dll since raw-dylib symbols from the current crate are passed to the linker
    // first, so any ambiguous names will prefer the current crate's definition.
    raw_dylib_test::inline_library_function();
    raw_dylib_test::library_function();
    raw_dylib_test_wrapper::inline_library_function_calls_inline();
    unsafe {
        extern_fn_2();
    }
}
