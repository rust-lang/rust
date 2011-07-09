// Unsafe pointer utility functions.

native "rust-intrinsic" mod rusti {
    fn ptr_offset[T](*T ptr, uint count) -> *T;
}

fn offset[T](*T ptr, uint count) -> *T { ret rusti::ptr_offset(ptr, count); }

