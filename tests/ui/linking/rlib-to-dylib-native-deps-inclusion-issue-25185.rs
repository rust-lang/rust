//@ run-pass
//@ aux-build:rlib-to-dylib-native-deps-inclusion-issue-25185-1.rs
//@ aux-build:rlib-to-dylib-native-deps-inclusion-issue-25185-2.rs

extern crate rlib_to_dylib_native_deps_inclusion_issue_25185_2 as minimal;

fn main() {
    let x = unsafe {
        minimal::rust_dbg_extern_identity_u32(1)
    };
    assert_eq!(x, 1);
}

// https://github.com/rust-lang/rust/issues/25185
