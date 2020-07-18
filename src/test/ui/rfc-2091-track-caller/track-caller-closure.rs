// run-pass

#![feature(stmt_expr_attributes)]

use std::panic::Location;

#[track_caller]
fn tracked() -> &'static Location<'static> {
    let get_location = #[track_caller] || Location::caller();
    get_location()
}

fn untracked_wrapper() -> (&'static Location<'static>, u32) {
    let get_location = #[track_caller] || Location::caller();
    (get_location(), line!())
}

fn nested_tracked() -> (&'static Location<'static>, u32) {
    (tracked(), line!())
}

fn main() {
    let get_location = #[track_caller] || Location::caller();
    let (location, line) = (get_location(), line!());
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), line);

    let (tracked, tracked_line) = (tracked(), line!());
    assert_eq!(tracked.file(), file!());
    assert_eq!(tracked.line(), tracked_line);

    let (nested, nested_line) = untracked_wrapper();
    assert_eq!(nested.file(), file!());
    assert_eq!(nested.line(), nested_line);

    let (contained, contained_line) = nested_tracked();
    assert_eq!(contained.file(), file!());
    assert_eq!(contained.line(), contained_line);

    fn pass_to_ptr_call<T, R>(f: fn(T) -> R, x: T) -> R {
        f(x)
    }

    let (get_location_w_n, line_from_shim) = (#[track_caller] |_| Location::caller(), line!());

    let (location_with_arg, line_with_arg) = (get_location_w_n(3), line!());
    assert_eq!(location_with_arg.file(), file!());
    assert_eq!(location_with_arg.line(), line_with_arg);

    let location_with_shim = pass_to_ptr_call(get_location_w_n, 5);
    // FIXME make the closure's "def site" point to this file
    assert_eq!(location_with_shim.file(), file!());
    assert_eq!(location_with_shim.line(), line_from_shim);
}
