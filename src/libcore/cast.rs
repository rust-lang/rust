//! Unsafe operations

export reinterpret_cast, forget, bump_box_refcount, transmute;
export transmute_mut, transmute_immut, transmute_region, transmute_mut_region;
export transmute_mut_unsafe, transmute_immut_unsafe;

export copy_lifetime, copy_lifetime_vec;

#[abi = "rust-intrinsic"]
extern mod rusti {
    #[legacy_exports];
    fn forget<T>(-x: T);
    fn reinterpret_cast<T, U>(e: T) -> U;
}

/// Casts the value at `src` to U. The two types must have the same length.
#[inline(always)]
unsafe fn reinterpret_cast<T, U>(src: &T) -> U {
    rusti::reinterpret_cast(*src)
}

/**
 * Move a thing into the void
 *
 * The forget function will take ownership of the provided value but neglect
 * to run any required cleanup or memory-management operations on it. This
 * can be used for various acts of magick, particularly when using
 * reinterpret_cast on managed pointer types.
 */
#[inline(always)]
unsafe fn forget<T>(-thing: T) { rusti::forget(move thing); }

/**
 * Force-increment the reference count on a shared box. If used
 * carelessly, this can leak the box. Use this in conjunction with transmute
 * and/or reinterpret_cast when such calls would otherwise scramble a box's
 * reference count
 */
unsafe fn bump_box_refcount<T>(+t: @T) { forget(move t); }

/**
 * Transform a value of one type into a value of another type.
 * Both types must have the same size and alignment.
 *
 * # Example
 *
 *     assert transmute("L") == ~[76u8, 0u8];
 */
#[inline(always)]
unsafe fn transmute<L, G>(-thing: L) -> G {
    let newthing: G = reinterpret_cast(&thing);
    forget(move thing);
    move newthing
}

/// Coerce an immutable reference to be mutable.
#[inline(always)]
unsafe fn transmute_mut<T>(+ptr: &a/T) -> &a/mut T { transmute(move ptr) }

/// Coerce a mutable reference to be immutable.
#[inline(always)]
unsafe fn transmute_immut<T>(+ptr: &a/mut T) -> &a/T { transmute(move ptr) }

/// Coerce a borrowed pointer to have an arbitrary associated region.
#[inline(always)]
unsafe fn transmute_region<T>(+ptr: &a/T) -> &b/T { transmute(move ptr) }

/// Coerce an immutable reference to be mutable.
#[inline(always)]
unsafe fn transmute_mut_unsafe<T>(+ptr: *const T) -> *mut T { transmute(ptr) }

/// Coerce an immutable reference to be mutable.
#[inline(always)]
unsafe fn transmute_immut_unsafe<T>(+ptr: *const T) -> *T { transmute(ptr) }

/// Coerce a borrowed mutable pointer to have an arbitrary associated region.
#[inline(always)]
unsafe fn transmute_mut_region<T>(+ptr: &a/mut T) -> &b/mut T {
    transmute(move ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline(always)]
unsafe fn copy_lifetime<S,T>(_ptr: &a/S, ptr: &T) -> &a/T {
    transmute_region(ptr)
}

/// Transforms lifetime of the second pointer to match the first.
#[inline(always)]
unsafe fn copy_lifetime_vec<S,T>(_ptr: &a/[S], ptr: &T) -> &a/T {
    transmute_region(ptr)
}


/****************************************************************************
 * Tests
 ****************************************************************************/

#[cfg(test)]
mod tests {
    #[legacy_exports];

    #[test]
    fn test_reinterpret_cast() {
        assert 1u == unsafe { reinterpret_cast(&1) };
    }

    #[test]
    fn test_bump_box_refcount() {
        unsafe {
            let box = @~"box box box";       // refcount 1
            bump_box_refcount(box);         // refcount 2
            let ptr: *int = transmute(box); // refcount 2
            let _box1: @~str = reinterpret_cast(&ptr);
            let _box2: @~str = reinterpret_cast(&ptr);
            assert *_box1 == ~"box box box";
            assert *_box2 == ~"box box box";
            // Will destroy _box1 and _box2. Without the bump, this would
            // use-after-free. With too many bumps, it would leak.
        }
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
            assert ~[76u8, 0u8] == transmute(~"L");
        }
    }
}
