#[doc = "Misc low level stuff"];

export type_desc;
export get_type_desc;
export size_of;
export min_align_of;
export pref_align_of;
export refcount;
export log_str;

enum type_desc = {
    first_param: **libc::c_int,
    size: libc::size_t,
    align: libc::size_t
    // Remaining fields not listed
};

#[abi = "cdecl"]
native mod rustrt {
    fn refcount<T>(t: @T) -> libc::intptr_t;
    fn unsupervise();
    fn shape_log_str<T>(t: *sys::type_desc, data: T) -> str;
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn get_tydesc<T>() -> *();
    fn size_of<T>() -> uint;
    fn pref_align_of<T>() -> uint;
    fn min_align_of<T>() -> uint;
}

#[doc = "
Returns a pointer to a type descriptor.

Useful for calling certain function in the Rust runtime or otherwise
performing dark magick.
"]
fn get_type_desc<T>() -> *type_desc {
    rusti::get_tydesc::<T>() as *type_desc
}

#[doc = "Returns the size of a type"]
fn size_of<T>() -> uint unsafe {
    rusti::size_of::<T>()
}

#[doc = "
Returns the ABI-required minimum alignment of a type

This is the alignment used for struct fields. It may be smaller
than the preferred alignment.
"]
fn min_align_of<T>() -> uint unsafe {
    rusti::min_align_of::<T>()
}

#[doc = "Returns the preferred alignment of a type"]
fn pref_align_of<T>() -> uint unsafe {
    rusti::pref_align_of::<T>()
}

#[doc = "Returns the refcount of a shared box"]
fn refcount<T>(t: @T) -> uint {
    ret rustrt::refcount::<T>(t) as uint;
}

fn log_str<T>(t: T) -> str {
    rustrt::shape_log_str(get_type_desc::<T>(), t)
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
