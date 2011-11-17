/*
Module: sys

Misc low level stuff
*/
tag type_desc {
    type_desc(@type_desc);
}

#[abi = "cdecl"]
native mod rustrt {
    // Explicitly re-export native stuff we want to be made
    // available outside this crate. Otherwise it's
    // visible-in-crate, but not re-exported.
    fn last_os_error() -> str;
    fn size_of(td: *type_desc) -> uint;
    fn align_of(td: *type_desc) -> uint;
    fn refcount<T>(t: @T) -> uint;
    fn do_gc();
    fn unsupervise();
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
    ret rustrt::last_os_error();
}

/*
Function: size_of

Returns the size of a type
*/
fn size_of<T>() -> uint {
    ret rustrt::size_of(get_type_desc::<T>());
}

/*
Function: align_of

Returns the alignment of a type
*/
fn align_of<T>() -> uint {
    ret rustrt::align_of(get_type_desc::<T>());
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

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
