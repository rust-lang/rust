// Tests that dlltool failing to generate an import library will raise an error.

// only-gnu
// only-windows
// needs-dlltool
// compile-flags: --crate-type lib --emit link
// normalize-stderr-test: "[^ ']*/dlltool.exe" -> "$$DLLTOOL"
// normalize-stderr-test: "[^ ]*/foo.def" -> "$$DEF_FILE"
#[link(name = "foo", kind = "raw-dylib")]
extern "C" {
    // `@1` is an invalid name to export, as it usually indicates that something
    // is being exported via ordinal.
    #[link_name = "@1"]
    fn f(x: i32);
}

pub fn lib_main() {
    unsafe { f(42); }
}
