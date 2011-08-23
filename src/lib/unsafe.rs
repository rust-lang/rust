// Unsafe operations.

native "rust-intrinsic" mod rusti {
    fn cast<T, U>(src: &T) -> U;
}

native "rust" mod rustrt {
    fn leak<@T>(thing: -T);
}

// Casts the value at `src` to U. The two types must have the same length.
fn reinterpret_cast<T, U>(src: &T) -> U { ret rusti::cast(src); }

fn leak<@T>(thing: -T) {
    rustrt::leak(thing);
}