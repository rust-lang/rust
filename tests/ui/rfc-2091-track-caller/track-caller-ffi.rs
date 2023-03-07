// run-pass

use std::panic::Location;

extern "Rust" {
    #[track_caller]
    fn rust_track_caller_ffi_test_tracked() -> &'static Location<'static>;
    fn rust_track_caller_ffi_test_untracked() -> &'static Location<'static>;
}

fn rust_track_caller_ffi_test_nested_tracked() -> &'static Location<'static> {
    unsafe { rust_track_caller_ffi_test_tracked() }
}

mod provides {
    use std::panic::Location;
    #[track_caller] // UB if we did not have this!
    #[no_mangle]
    fn rust_track_caller_ffi_test_tracked() -> &'static Location<'static> {
        Location::caller()
    }
    #[no_mangle]
    fn rust_track_caller_ffi_test_untracked() -> &'static Location<'static> {
        Location::caller()
    }
}

fn main() {
    let location = Location::caller();
    assert_eq!(location.file(), file!());
    assert_eq!(location.line(), 29);
    assert_eq!(location.column(), 20);

    let tracked = unsafe { rust_track_caller_ffi_test_tracked() };
    assert_eq!(tracked.file(), file!());
    assert_eq!(tracked.line(), 34);
    assert_eq!(tracked.column(), 28);

    let untracked = unsafe { rust_track_caller_ffi_test_untracked() };
    assert_eq!(untracked.file(), file!());
    assert_eq!(untracked.line(), 24);
    assert_eq!(untracked.column(), 9);

    let contained = rust_track_caller_ffi_test_nested_tracked();
    assert_eq!(contained.file(), file!());
    assert_eq!(contained.line(), 12);
    assert_eq!(contained.column(), 14);
}
