//@ only-x86_64
//@ only-windows
//@ compile-flags: --crate-type lib --emit link
#[link(name = "foo", kind = "raw-dylib")]
extern "stdcall" {
//~^ WARN: calling convention not supported on this target
//~| WARN: previously accepted
    fn f(x: i32);
    //~^ ERROR ABI not supported by `#[link(kind = "raw-dylib")]` on this architecture
}

pub fn lib_main() {
    unsafe { f(42); }
}
