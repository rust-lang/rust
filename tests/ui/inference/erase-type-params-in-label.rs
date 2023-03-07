fn main() {
    let foo = foo(1, ""); //~ ERROR E0283
}
fn baz() {
    let bar = bar(1, ""); //~ ERROR E0283
}

struct Bar<T, K, N: Default> {
    t: T,
    k: K,
    n: N,
}

fn bar<T, K, Z: Default>(t: T, k: K) -> Bar<T, K, Z> {
    Bar { t, k, n: Default::default() }
}

struct Foo<T, K, N: Default, M: Default> {
    t: T,
    k: K,
    n: N,
    m: M,
}

fn foo<T, K, W: Default, Z: Default>(t: T, k: K) -> Foo<T, K, W, Z> {
    Foo { t, k, n: Default::default(), m: Default::default() }
}
