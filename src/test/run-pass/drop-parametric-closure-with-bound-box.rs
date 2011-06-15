

fn f[T](@int i, &T t) { }

fn main() { auto x = bind f[char](@0xdeafbeef, _); }