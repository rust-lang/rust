// Unsafe operations.

native "rust-intrinsic" mod rusti {
    fn cast[T, U](src: &T) -> U;
}

// Casts the value at `src` to U. The two types must have the same length.
fn reinterpret_cast[T, U](src: &T) -> U { ret rusti::cast(src); }

