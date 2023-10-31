// Tests that failing to run dlltool will raise an error.

// only-gnu
// only-windows
// compile-flags: --crate-type lib --emit link -Cdlltool=does_not_exit.exe
#[link(name = "foo", kind = "raw-dylib")]
extern "C" {
    fn f(x: i32);
}

pub fn lib_main() {
    unsafe { f(42); }
}
