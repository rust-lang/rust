// ignore-arm stdcall isn't supported
#![feature(extended_varargs_abi_support)]

fn baz(f: extern "stdcall" fn(usize, ...)) {
    //~^ ERROR: C-variadic function must have a compatible calling convention,
    // like C, cdecl, aapcs, win64, sysv64 or efiapi
    f(22, 44);
}

fn aapcs(f: extern "aapcs" fn(usize, ...)) {
    f(22, 44);
}
fn sysv(f: extern "sysv64" fn(usize, ...)) {
    f(22, 44);
}
fn win(f: extern "win64" fn(usize, ...)) {
    f(22, 44);
}
fn efiapi(f: extern "efiapi" fn(usize, ...)) {
    f(22, 44);
}

fn main() {}
