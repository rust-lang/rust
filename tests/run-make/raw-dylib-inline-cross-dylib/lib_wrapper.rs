extern crate raw_dylib_test;

#[inline]
pub fn inline_library_function_calls_inline() {
    raw_dylib_test::inline_library_function();
}
