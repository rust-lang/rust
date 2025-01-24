//! Check that crates can be linked together with `-Z sanitizer=address` on msvc.
//! See <https://github.com/rust-lang/rust/issues/124390>.

//@ run-pass
//@ compile-flags:-Zsanitizer=address
//@ aux-build: asan_odr_win-2.rs
//@ only-windows-msvc

extern crate othercrate;

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("hello!");
    });
    assert!(result.is_ok());

    othercrate::exposed_func();
}
