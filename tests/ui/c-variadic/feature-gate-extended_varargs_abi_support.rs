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

extern "efiapi" {
    fn extern_efiapi(...);
    //~^ ERROR using calling conventions other than `C` or `cdecl` for varargs functions is unstable [E0658]
}

extern "sysv64" {
    fn extern_sysv64(...);
    //~^ ERROR using calling conventions other than `C` or `cdecl` for varargs functions is unstable [E0658]
}

extern "win64" {
    fn extern_win64(...);
    //~^ ERROR using calling conventions other than `C` or `cdecl` for varargs functions is unstable [E0658]
}

fn main() {}
