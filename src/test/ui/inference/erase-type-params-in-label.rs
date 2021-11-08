fn main() {
    let foo = new(1, ""); //~ ERROR E0283
}

struct Bar<T, K, N: Default> {
    t: T,
    k: K,
    n: N,
}

fn new<T, K, Z: Default>(t: T, k: K) -> Bar<T, K, Z> {
    Bar { t, k, n: Default::default() }
}
