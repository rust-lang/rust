// Unsafe pointer utility functions.

native "rust-intrinsic" mod rusti {
    fn addr_of<T>(val: T) -> *mutable T;
    fn ptr_offset<T>(ptr: *T, count: uint) -> *T;
}

fn addr_of<@T>(val: T) -> *mutable T { ret rusti::addr_of(val); }
fn offset<@T>(ptr: *T, count: uint) -> *T {
    ret rusti::ptr_offset(ptr, count);
}

fn null<@T>() -> *T { ret unsafe::reinterpret_cast(0u); }
