#![allow(unused_macros)]
#![allow(unused_macro_rules)]
#![allow(unused_imports)]

/// `alloc_id!(ptr)`: obtain the allocation id from a pointer.
///
/// `ptr` should be any pointer or reference that can be converted with `_ as *const u8`.
///
/// The id obtained can be passed directly to `print_state!`.
macro_rules! alloc_id {
    ($ptr:expr) => {
        $crate::utils::miri_get_alloc_id($ptr as *const u8 as *const ())
    };
}

/// `print_state!(alloc_id, show_unnamed)`: print the internal state of the borrow
/// tracker (stack or tree).
///
/// `alloc_id` should be obtained from `alloc_id!`.
///
/// `show_unnamed` is an optional boolean that determines if Tree Borrows displays
/// tags that have not been given a name. Defaults to `false`.
macro_rules! print_state {
    ($alloc_id:expr) => {
        print_state!($alloc_id, false);
    };
    ($alloc_id:expr, $show:expr) => {
        $crate::utils::miri_print_borrow_state($alloc_id, $show);
    };
}

/// `name!(ptr => nth_parent, name)`: associate `name` to the `nth_parent` of `ptr`.
///
/// `ptr` should be any pointer or reference that can be converted with `_ as *const u8`.
///
/// `nth_parent` is an optional `u8` that defaults to 0. The corresponding ancestor
/// of the tag of `ptr` will be searched: 0 for `ptr` itself, 1 for the direct parent
/// of `ptr`, 2 for the grandparent, etc. If `nth_parent` is not specified,
/// then `=>` should also not be included.
///
/// `name` is an optional string that will be used as the name. Defaults to
/// `stringify!($ptr)` the name of `ptr` in the source code.
macro_rules! name {
    ($ptr:expr, $name:expr) => {
        name!($ptr => 0, $name);
    };
    ($ptr:expr) => {
        name!($ptr => 0, stringify!($ptr));
    };
    ($ptr:expr => $nth_parent:expr) => {
        name!($ptr => $nth_parent, stringify!($ptr));
    };
    ($ptr:expr => $nth_parent:expr, $name:expr) => {
        let name = $name.as_bytes();
        $crate::utils::miri_pointer_name($ptr as *const u8 as *const (), $nth_parent, name);
    };
}
