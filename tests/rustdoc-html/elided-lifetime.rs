//@ aux-build:elided-lifetime.rs
//
// rust-lang/rust#75225
//
// Since Rust 2018 we encourage writing out <'_> explicitly to make it clear
// that borrowing is occurring. Make sure rustdoc is following the same idiom.

#![crate_name = "foo"]

pub struct Ref<'a>(&'a u32);
type ARef<'a> = Ref<'a>;

//@ has foo/fn.test1.html
//@ matchesraw - "Ref</a>&lt;'_&gt;"
pub fn test1(a: &u32) -> Ref {
    Ref(a)
}

//@ has foo/fn.test2.html
//@ matchesraw - "Ref</a>&lt;'_&gt;"
pub fn test2(a: &u32) -> Ref<'_> {
    Ref(a)
}

//@ has foo/fn.test3.html
//@ matchesraw - "Ref</a>&lt;'_&gt;"
pub fn test3(a: &u32) -> ARef {
    Ref(a)
}

//@ has foo/fn.test4.html
//@ matchesraw - "Ref</a>&lt;'_&gt;"
pub fn test4(a: &u32) -> ARef<'_> {
    Ref(a)
}

// Ensure external paths in inlined docs also display elided lifetime
//@ has foo/bar/fn.test5.html
//@ matchesraw - "Ref</a>&lt;'_&gt;"
//@ has foo/bar/fn.test6.html
//@ matchesraw - "Ref</a>&lt;'_&gt;"
#[doc(inline)]
pub extern crate bar;
