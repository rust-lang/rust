#[doc = "Unsafe operations"];

export reinterpret_cast, forget, transmute;

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

#[doc = "
Transform a value of one type into a value of another type.
Both types must have the same size and alignment.

# Example

    assert transmute(\"L\") == [76u8, 0u8]/~;
"]
unsafe fn transmute<L, G>(-thing: L) -> G {
    let newthing = reinterpret_cast(thing);
    forget(thing);
    ret newthing;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_reinterpret_cast() {
        assert unsafe { reinterpret_cast(1) } == 1u;
    }

    #[test]
    fn test_transmute() {
        unsafe {
            let x = @1;
            let x: *int = transmute(x);
            assert *x == 1;
            let _x: @int = transmute(x);
        }
    }

    #[test]
    fn test_transmute2() {
        unsafe {
            assert transmute("L") == [76u8, 0u8]/~;
        }
    }
}
