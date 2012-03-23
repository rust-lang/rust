#[doc = "Unsafe operations"];

export reinterpret_cast, forget;

#[abi = "rust-intrinsic"]
native mod rusti {
    fn forget<T>(-x: T);
    fn reinterpret_cast<T, U>(e: T) -> U;
}

#[doc = "
Casts the value at `src` to U. The two types must have the same length.
"]
#[inline(always)]
unsafe fn reinterpret_cast<T, U>(src: T) -> U {
    rusti::reinterpret_cast(src)
}

#[doc ="
Move a thing into the void

The forget function will take ownership of the provided value but neglect
to run any required cleanup or memory-management operations on it. This
can be used for various acts of magick, particularly when using
reinterpret_cast on managed pointer types.
"]
#[inline(always)]
unsafe fn forget<T>(-thing: T) { rusti::forget(thing); }

#[cfg(test)]
mod tests {

    #[test]
    fn test_reinterpret_cast() unsafe {
        assert reinterpret_cast(1) == 1u;
    }
}
