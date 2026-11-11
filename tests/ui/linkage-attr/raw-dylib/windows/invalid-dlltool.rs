// Tests that failing to run dlltool will raise an error.

//@ needs-dlltool
//@ compile-flags: --crate-type lib --emit link -Cdlltool=does_not_exist.exe
#[link(name = "foo", kind = "raw-dylib")]
extern "C" {
    fn f(x: i32);
}

pub fn lib_main() {
    unsafe { f(42); }
}

//~? ERROR error calling dlltool 'does_not_exist.exe': program not found
