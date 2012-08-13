//! Misc low level stuff

export TypeDesc;
export get_type_desc;
export size_of;
export min_align_of;
export pref_align_of;
export refcount;
export log_str;
export shape_eq, shape_lt, shape_le;

// Corresponds to runtime type_desc type
enum TypeDesc = {
    size: uint,
    align: uint
    // Remaining fields not listed
};

#[abi = "cdecl"]
extern mod rustrt {
    pure fn shape_log_str(t: *sys::TypeDesc, data: *()) -> ~str;
}

#[abi = "rust-intrinsic"]
extern mod rusti {
    fn get_tydesc<T>() -> *();
    fn size_of<T>() -> uint;
    fn pref_align_of<T>() -> uint;
    fn min_align_of<T>() -> uint;
}

/// Compares contents of two pointers using the default method.
/// Equivalent to `*x1 == *x2`.  Useful for hashtables.
pure fn shape_eq<T>(x1: &T, x2: &T) -> bool {
    *x1 == *x2
}

pure fn shape_lt<T>(x1: &T, x2: &T) -> bool {
    *x1 < *x2
}

pure fn shape_le<T>(x1: &T, x2: &T) -> bool {
    *x1 < *x2
}

/**
 * Returns a pointer to a type descriptor.
 *
 * Useful for calling certain function in the Rust runtime or otherwise
 * performing dark magick.
 */
pure fn get_type_desc<T>() -> *TypeDesc {
    unchecked { rusti::get_tydesc::<T>() as *TypeDesc }
}

/// Returns the size of a type
#[inline(always)]
pure fn size_of<T>() -> uint {
    unchecked { rusti::size_of::<T>() }
}

/**
 * Returns the ABI-required minimum alignment of a type
 *
 * This is the alignment used for struct fields. It may be smaller
 * than the preferred alignment.
 */
pure fn min_align_of<T>() -> uint {
    unchecked { rusti::min_align_of::<T>() }
}

/// Returns the preferred alignment of a type
pure fn pref_align_of<T>() -> uint {
    unchecked { rusti::pref_align_of::<T>() }
}

/// Returns the refcount of a shared box (as just before calling this)
pure fn refcount<T>(+t: @T) -> uint {
    unsafe {
        let ref_ptr: *uint = unsafe::reinterpret_cast(t);
        *ref_ptr - 1
    }
}

pure fn log_str<T>(t: T) -> ~str {
    unsafe {
        let data_ptr: *() = unsafe::reinterpret_cast(ptr::addr_of(t));
        rustrt::shape_log_str(get_type_desc::<T>(), data_ptr)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn size_of_basic() {
        assert size_of::<u8>() == 1u;
        assert size_of::<u16>() == 2u;
        assert size_of::<u32>() == 4u;
        assert size_of::<u64>() == 8u;
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    fn size_of_32() {
        assert size_of::<uint>() == 4u;
        assert size_of::<*uint>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn size_of_64() {
        assert size_of::<uint>() == 8u;
        assert size_of::<*uint>() == 8u;
    }

    #[test]
    fn align_of_basic() {
        assert pref_align_of::<u8>() == 1u;
        assert pref_align_of::<u16>() == 2u;
        assert pref_align_of::<u32>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    fn align_of_32() {
        assert pref_align_of::<uint>() == 4u;
        assert pref_align_of::<*uint>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn align_of_64() {
        assert pref_align_of::<uint>() == 8u;
        assert pref_align_of::<*uint>() == 8u;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
