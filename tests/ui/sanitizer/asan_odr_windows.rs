//! Check that crates can be linked together with `-Z sanitizer=address` on msvc.
//! See <https://github.com/rust-lang/rust/issues/124390>.

//@ ignore-test (FIXME #140189, cannot open file 'clang_rt.asan_dynamic_runtime_thunk-x86_64.lib')

// FIXME @ run-pass
// FIXME @ compile-flags:-Zsanitizer=address
// FIXME @ aux-build: asan_odr_win-2.rs
// FIXME @ only-windows-msvc

extern crate othercrate;

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("hello!");
    });
    assert!(result.is_ok());

    othercrate::exposed_func();
}
