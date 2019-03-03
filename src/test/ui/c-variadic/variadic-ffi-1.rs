// ignore-arm stdcall isn't supported
// ignore-aarch64 stdcall isn't supported

extern "stdcall" {
    fn printf(_: *const u8, ...); //~ ERROR: variadic function must have C or cdecl calling
}

extern {
    fn foo(f: isize, x: u8, ...);
}

extern "C" fn bar(f: isize, x: u8) {}

fn main() {
    // errors below are no longer checked because error above aborts
    // compilation; see variadic-ffi-3.rs for corresponding test.
    unsafe {
        foo();
        foo(1);

        let x: unsafe extern "C" fn(f: isize, x: u8) = foo;
        let y: extern "C" fn(f: isize, x: u8, ...) = bar;

        foo(1, 2, 3f32);
        foo(1, 2, true);
        foo(1, 2, 1i8);
        foo(1, 2, 1u8);
        foo(1, 2, 1i16);
        foo(1, 2, 1u16);
    }
}
