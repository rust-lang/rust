//! Misc low level stuff

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cmp::{Eq, Ord};
use libc::c_void;

pub type FreeGlue = fn(*TypeDesc, *c_void);

// Corresponds to runtime type_desc type
pub enum TypeDesc = {
    size: uint,
    align: uint,
    take_glue: uint,
    drop_glue: uint,
    free_glue: uint
    // Remaining fields not listed
};

/// The representation of a Rust closure
pub struct Closure {
    code: *(),
    env: *(),
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
pub pure fn shape_eq<T:Eq>(x1: &T, x2: &T) -> bool {
    *x1 == *x2
}

pub pure fn shape_lt<T:Ord>(x1: &T, x2: &T) -> bool {
    *x1 < *x2
}

pub pure fn shape_le<T:Ord>(x1: &T, x2: &T) -> bool {
    *x1 <= *x2
}

/**
 * Returns a pointer to a type descriptor.
 *
 * Useful for calling certain function in the Rust runtime or otherwise
 * performing dark magick.
 */
#[inline(always)]
pub pure fn get_type_desc<T>() -> *TypeDesc {
    unsafe { rusti::get_tydesc::<T>() as *TypeDesc }
}

/// Returns the size of a type
#[inline(always)]
pub pure fn size_of<T>() -> uint {
    unsafe { rusti::size_of::<T>() }
}

/**
 * Returns the ABI-required minimum alignment of a type
 *
 * This is the alignment used for struct fields. It may be smaller
 * than the preferred alignment.
 */
#[inline(always)]
pub pure fn min_align_of<T>() -> uint {
    unsafe { rusti::min_align_of::<T>() }
}

/// Returns the preferred alignment of a type
#[inline(always)]
pub pure fn pref_align_of<T>() -> uint {
    unsafe { rusti::pref_align_of::<T>() }
}

/// Returns the refcount of a shared box (as just before calling this)
#[inline(always)]
pub pure fn refcount<T>(+t: @T) -> uint {
    unsafe {
        let ref_ptr: *uint = cast::reinterpret_cast(&t);
        *ref_ptr - 1
    }
}

pub pure fn log_str<T>(t: &T) -> ~str {
    unsafe {
        do io::with_str_writer |wr| {
            repr::write_repr(wr, t)
        }
    }
}

#[cfg(test)]
pub mod tests {

    #[test]
    pub fn size_of_basic() {
        assert size_of::<u8>() == 1u;
        assert size_of::<u16>() == 2u;
        assert size_of::<u32>() == 4u;
        assert size_of::<u64>() == 8u;
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    pub fn size_of_32() {
        assert size_of::<uint>() == 4u;
        assert size_of::<*uint>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    pub fn size_of_64() {
        assert size_of::<uint>() == 8u;
        assert size_of::<*uint>() == 8u;
    }

    #[test]
    pub fn align_of_basic() {
        assert pref_align_of::<u8>() == 1u;
        assert pref_align_of::<u16>() == 2u;
        assert pref_align_of::<u32>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    pub fn align_of_32() {
        assert pref_align_of::<uint>() == 4u;
        assert pref_align_of::<*uint>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    pub fn align_of_64() {
        assert pref_align_of::<uint>() == 8u;
        assert pref_align_of::<*uint>() == 8u;
    }

    #[test]
    pub fn synthesize_closure() unsafe {
        let x = 10;
        let f: fn(int) -> int = |y| x + y;

        assert f(20) == 30;

        let original_closure: Closure = cast::transmute(f);

        let actual_function_pointer = original_closure.code;
        let environment = original_closure.env;

        let new_closure = Closure {
            code: actual_function_pointer,
            env: environment
        };

        let new_f: fn(int) -> int = cast::transmute(new_closure);
        assert new_f(20) == 30;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
