//@ known-bug: rust-lang/rust#146834
//@compile-flags: -Wsingle-use-lifetimes
#![core::contracts::ensures]

fn f4_<'a, 'b>(a: &'a i32, b: &'b i32) -> (&i32, &i32) {
    loop {}
}
