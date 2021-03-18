// run-pass
// ignore-windows
// ignore-macos
// ignore-emscripten doesn't support this linkage
// ignore-sgx weak linkage not permitted
// aux-build:linkage1.rs

#![feature(linkage)]

extern crate linkage1 as other;

extern "C" {
    #[linkage = "extern_weak"]
    static foo: *const isize;
    #[linkage = "extern_weak"]
    static something_that_should_never_exist: *mut isize;
}

fn main() {
    // It appears that the --as-needed flag to linkers will not pull in a dynamic
    // library unless it satisfies a non weak undefined symbol. The 'other' crate
    // is compiled as a dynamic library where it would only be used for a
    // weak-symbol as part of an executable, so the dynamic library would be
    // discarded. By adding and calling `other::bar`, we get around this problem.
    other::bar();

    unsafe {
        assert!(!foo.is_null());
        assert_eq!(*foo, 3);
        assert!(something_that_should_never_exist.is_null());
    }
}
