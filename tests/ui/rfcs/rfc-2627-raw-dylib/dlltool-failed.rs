// Tests that dlltool failing to generate an import library will raise an error.

//@ needs-dlltool
//@ compile-flags: --crate-type lib --emit link
//@ normalize-stderr-test: "[^ ']*/dlltool.exe" -> "$$DLLTOOL"
//@ normalize-stderr-test: "[^ ]*/foo.dll_imports.def" -> "$$DEF_FILE"
//@ normalize-stderr-test: "[^ ]*/foo.dll_imports.lib" -> "$$LIB_FILE"
//@ normalize-stderr-test: "-m [^ ]*" -> "$$TARGET_MACHINE"
//@ normalize-stderr-test: "-f [^ ]*" -> "$$ASM_FLAGS"
//@ normalize-stderr-test: "--temp-prefix [^ ]*/foo.dll" -> "$$TEMP_PREFIX"
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
