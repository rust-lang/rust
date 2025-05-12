//@ only-x86_64

fn efiapi(f: extern "efiapi" fn(usize, ...)) {
    //~^ ERROR: using calling conventions other than `C` or `cdecl` for varargs functions is unstable
    f(22, 44);
}
fn sysv(f: extern "sysv64" fn(usize, ...)) {
    //~^ ERROR: using calling conventions other than `C` or `cdecl` for varargs functions is unstable
    f(22, 44);
}
fn win(f: extern "win64" fn(usize, ...)) {
    //~^ ERROR: using calling conventions other than `C` or `cdecl` for varargs functions is unstable
    f(22, 44);
}

fn main() {}
