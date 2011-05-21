export ptr_eq;

native "rust" mod rustrt {
    fn rust_ptr_eq[T](@T a, @T b) -> int;
}

fn ptr_eq[T](@T a, @T b) -> bool { ret rustrt::rust_ptr_eq[T](a, b) != 0; }

