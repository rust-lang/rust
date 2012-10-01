#[forbid(deprecated_mode)];
//! Unsafe debugging functions for inspecting values.

use cast::reinterpret_cast;


#[abi = "cdecl"]
extern mod rustrt {
    #[legacy_exports];
    fn debug_tydesc(td: *sys::TypeDesc);
    fn debug_opaque(td: *sys::TypeDesc, x: *());
    fn debug_box(td: *sys::TypeDesc, x: *());
    fn debug_tag(td: *sys::TypeDesc, x: *());
    fn debug_fn(td: *sys::TypeDesc, x: *());
    fn debug_ptrcast(td: *sys::TypeDesc, x: *()) -> *();
    fn rust_dbg_breakpoint();
}

pub fn debug_tydesc<T>() {
    rustrt::debug_tydesc(sys::get_type_desc::<T>());
}

pub fn debug_opaque<T>(+x: T) {
    rustrt::debug_opaque(sys::get_type_desc::<T>(), ptr::addr_of(&x) as *());
}

pub fn debug_box<T>(x: @T) {
    rustrt::debug_box(sys::get_type_desc::<T>(), ptr::addr_of(&x) as *());
}

pub fn debug_tag<T>(+x: T) {
    rustrt::debug_tag(sys::get_type_desc::<T>(), ptr::addr_of(&x) as *());
}

pub fn debug_fn<T>(+x: T) {
    rustrt::debug_fn(sys::get_type_desc::<T>(), ptr::addr_of(&x) as *());
}

pub unsafe fn ptr_cast<T, U>(x: @T) -> @U {
    reinterpret_cast(
        &rustrt::debug_ptrcast(sys::get_type_desc::<T>(),
                              reinterpret_cast(&x)))
}

/// Triggers a debugger breakpoint
pub fn breakpoint() {
    rustrt::rust_dbg_breakpoint();
}

#[test]
fn test_breakpoint_should_not_abort_process_when_not_under_gdb() {
    // Triggering a breakpoint involves raising SIGTRAP, which terminates
    // the process under normal circumstances
    breakpoint();
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
