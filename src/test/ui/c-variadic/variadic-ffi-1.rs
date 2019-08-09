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
    unsafe {
        foo();  //~ ERROR this function takes at least 2 parameters but 0 parameters were supplied
        foo(1); //~ ERROR this function takes at least 2 parameters but 1 parameter was supplied

        let x: unsafe extern "C" fn(f: isize, x: u8) = foo; //~ ERROR mismatched types
        let y: extern "C" fn(f: isize, x: u8, ...) = bar; //~ ERROR mismatched types

        foo(1, 2, 3f32); //~ ERROR can't pass
        foo(1, 2, true); //~ ERROR can't pass
        foo(1, 2, 1i8);  //~ ERROR can't pass
        foo(1, 2, 1u8);  //~ ERROR can't pass
        foo(1, 2, 1i16); //~ ERROR can't pass
        foo(1, 2, 1u16); //~ ERROR can't pass
    }
}
