#[doc = "Unsafe debugging functions for inspecting values."];

#[abi = "cdecl"]
native mod rustrt {
    fn debug_tydesc(td: *sys::type_desc);
    fn debug_opaque<T>(td: *sys::type_desc, x: T);
    fn debug_box<T>(td: *sys::type_desc, x: @T);
    fn debug_tag<T>(td: *sys::type_desc, x: T);
    fn debug_fn<T>(td: *sys::type_desc, x: T);
    fn debug_ptrcast<T, U>(td: *sys::type_desc, x: @T) -> @U;
}

fn debug_tydesc<T>() {
    rustrt::debug_tydesc(sys::get_type_desc::<T>());
}

fn debug_opaque<T>(x: T) {
    rustrt::debug_opaque::<T>(sys::get_type_desc::<T>(), x);
}

fn debug_box<T>(x: @T) {
    rustrt::debug_box::<T>(sys::get_type_desc::<T>(), x);
}

fn debug_tag<T>(x: T) {
    rustrt::debug_tag::<T>(sys::get_type_desc::<T>(), x);
}

fn debug_fn<T>(x: T) {
    rustrt::debug_fn::<T>(sys::get_type_desc::<T>(), x);
}

unsafe fn ptr_cast<T, U>(x: @T) -> @U {
    ret rustrt::debug_ptrcast::<T, U>(sys::get_type_desc::<T>(), x);
}

fn refcount<T>(a: @T) -> uint unsafe {
    let p: *uint = unsafe::reinterpret_cast(a);
    ret *p;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
