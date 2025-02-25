fn system(f: extern "system" fn(usize, ...)) {
    //~^  ERROR using calling conventions other than `C` or `cdecl` for varargs functions is unstable

    f(22, 44);
}

fn main() {}
