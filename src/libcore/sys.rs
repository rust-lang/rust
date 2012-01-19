/*
Module: sys

Misc low level stuff
*/
enum type_desc = {
    first_param: **ctypes::c_int,
    size: ctypes::size_t,
    align: ctypes::size_t
    // Remaining fields not listed
};

#[abi = "cdecl"]
native mod rustrt {
    // Explicitly re-export native stuff we want to be made
    // available outside this crate. Otherwise it's
    // visible-in-crate, but not re-exported.
    fn last_os_error() -> str;
    fn refcount<T>(t: @T) -> ctypes::intptr_t;
    fn do_gc();
    fn unsupervise();
    fn shape_log_str<T>(t: *sys::type_desc, data: T) -> str;
    fn rust_set_exit_status(code: ctypes::intptr_t);
    fn set_min_stack(size: ctypes::uintptr_t);
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn get_type_desc<T>() -> *type_desc;
}

/*
Function: get_type_desc

Returns a pointer to a type descriptor. Useful for calling certain
function in the Rust runtime or otherwise performing dark magick.
*/
fn get_type_desc<T>() -> *type_desc {
    ret rusti::get_type_desc::<T>();
}

/*
Function: last_os_error

Get a string representing the platform-dependent last error
*/
fn last_os_error() -> str {
    rustrt::last_os_error()
}

/*
Function: size_of

Returns the size of a type
*/
fn size_of<T>() -> uint unsafe {
    ret (*get_type_desc::<T>()).size;
}

/*
Function: align_of

Returns the alignment of a type
*/
fn align_of<T>() -> uint unsafe {
    ret (*get_type_desc::<T>()).align;
}

/*
Function: refcount

Returns the refcount of a shared box
*/
fn refcount<T>(t: @T) -> uint {
    ret rustrt::refcount::<T>(t);
}

/*
Function: do_gc

Force a garbage collection
*/
fn do_gc() -> () {
    ret rustrt::do_gc();
}

// FIXME: There's a wrapper for this in the task module and this really
// just belongs there
fn unsupervise() -> () {
    ret rustrt::unsupervise();
}

fn log_str<T>(t: T) -> str {
    rustrt::shape_log_str(get_type_desc::<T>(), t)
}

#[doc(
    brief = "Sets the process exit code",
    desc = "Sets the exit code returned by the process if all supervised \
            tasks terminate successfully (without failing). If the current \
            root task fails and is supervised by the scheduler then any \
            user-specified exit status is ignored and the process exits \
            with the default failure status."
)]
fn set_exit_status(code: int) {
    rustrt::rust_set_exit_status(code as ctypes::intptr_t);
}

// FIXME: #1495 - This shouldn't exist
#[doc(
    brief =
    "Globally set the minimum size, in bytes, of a stack segment",
    desc =
    "Rust tasks have segmented stacks that are connected in a linked list \
     allowing them to start very small and grow very large. In some \
     situations this can result in poor performance. Calling this function \
     will set the minimum size of all stack segments allocated in the \
     future, for all tasks."
)]
#[deprecated]
fn set_min_stack(size: uint) {
    rustrt::set_min_stack(size);
}

#[cfg(test)]
mod tests {

    #[test]
    fn last_os_error() {
        log(debug, last_os_error());
    }

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
        assert align_of::<u8>() == 1u;
        assert align_of::<u16>() == 2u;
        assert align_of::<u32>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86")]
    #[cfg(target_arch = "arm")]
    fn align_of_32() {
        assert align_of::<uint>() == 4u;
        assert align_of::<*uint>() == 4u;
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn align_of_64() {
        assert align_of::<uint>() == 8u;
        assert align_of::<*uint>() == 8u;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
