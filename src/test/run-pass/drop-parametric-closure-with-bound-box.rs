

fn f<T>(i: @int, t: &T) { }

fn main() { let x = bind f[char](@0xdeafbeef, _); }
