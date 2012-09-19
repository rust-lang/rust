// xfail-fast
#[legacy_modes];

fn region_identity(x: &r/uint) -> &r/uint { x }

fn apply<T>(t: T, f: fn(T) -> T) -> T { f(t) }

fn parameterized(x: &uint) -> uint {
    let z = apply(x, ({|y|
        region_identity(y)
    }));
    *z
}

fn main() {
    let x = 3u;
    assert parameterized(&x) == 3u;
}
