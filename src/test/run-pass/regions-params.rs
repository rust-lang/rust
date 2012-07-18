fn region_identity(x: &uint) -> &uint { x }

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