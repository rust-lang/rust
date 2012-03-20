#[doc = "Unsafe operations"];

export reinterpret_cast, forget;

#[abi = "rust-intrinsic"]
native mod rusti {
    fn cast<T, U>(src: T) -> U;
    fn leak<T>(-thing: T);
}

#[doc = "
Casts the value at `src` to U. The two types must have the same length.
"]
#[inline(always)]
unsafe fn reinterpret_cast<T, U>(src: T) -> U {
    let t1 = sys::get_type_desc::<T>();
    let t2 = sys::get_type_desc::<U>();
    if (*t1).size != (*t2).size {
        fail "attempt to cast values of differing sizes";
    }
    ret rusti::cast(src);
}

#[doc ="
Move a thing into the void

The forget function will take ownership of the provided value but neglect
to run any required cleanup or memory-management operations on it. This
can be used for various acts of magick, particularly when using
reinterpret_cast on managed pointer types.
"]
#[inline(always)]
unsafe fn forget<T>(-thing: T) { rusti::leak(thing); }

#[cfg(test)]
mod tests {

    #[test]
    fn test_reinterpret_cast() unsafe {
        assert reinterpret_cast(1) == 1u;
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win32"))]
    fn test_reinterpret_cast_wrong_size() unsafe {
        let _i: uint = reinterpret_cast(0u8);
    }
}
