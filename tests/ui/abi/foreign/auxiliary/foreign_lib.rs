#![crate_name = "foreign_lib"]

pub mod rustrt {
    #[link(name = "rust_test_helpers", kind = "static")]
    extern "C" {
        pub fn rust_get_test_int() -> isize;
    }
}

pub mod rustrt2 {
    extern "C" {
        pub fn rust_get_test_int() -> isize;
    }
}

pub mod rustrt3 {
    // The point of this test is to ensure that we don't ICE or trigger LLVM asserts when importing
    // the same symbol with different types. This is not really possible to test portably; there is
    // no different signature we can come up with that is different to LLVM but which for sure has
    // the same behavior on all platforms. The signed-ness of integers is ignored by LLVM as well
    // as pointee types. So the only ways to make our signatures differ are to use
    // differently-sized integers which is definitely an ABI mismatch, or to rely on pointers and
    // isize/usize having the same ABI, which is wrong on CHERI and probably other niche platforms.
    // If this test causes you trouble, please file an issue.
    // See https://github.com/rust-lang/rust/issues/32740 for the bug that prompted this test.
    extern "C" {
        pub fn rust_get_test_int() -> *const u8;
    }
}

pub fn local_uses() {
    unsafe {
        let x = rustrt::rust_get_test_int();
        assert_eq!(x, rustrt2::rust_get_test_int());
        assert_eq!(x as *const u8, rustrt3::rust_get_test_int());
    }
}
