// ignore-arm stdcall isn't supported

fn baz(f: extern "stdcall" fn(usize, ...)) {
    //~^ ERROR: variadic function must have C or cdecl calling convention
    f(22, 44);
}

fn main() {}
