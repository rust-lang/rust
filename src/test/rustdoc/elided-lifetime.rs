#![crate_name = "foo"]

// rust-lang/rust#75225
//
// Since Rust 2018 we encourage writing out <'_> explicitly to make it clear
// that borrowing is occuring. Make sure rustdoc is following the same idiom.

pub struct Ref<'a>(&'a u32);
type ARef<'a> = Ref<'a>;

// @has foo/fn.test1.html
// @matches - "Ref</a>&lt;'_&gt;"
pub fn test1(a: &u32) -> Ref {
    Ref(a)
}

// @has foo/fn.test2.html
// @matches - "Ref</a>&lt;'_&gt;"
pub fn test2(a: &u32) -> Ref<'_> {
    Ref(a)
}

// @has foo/fn.test3.html
// @matches - "Ref</a>&lt;'_&gt;"
pub fn test3(a: &u32) -> ARef {
    Ref(a)
}

// @has foo/fn.test4.html
// @matches - "Ref</a>&lt;'_&gt;"
pub fn test4(a: &u32) -> ARef<'_> {
    Ref(a)
}

// Ensure external paths also display elided lifetime
// @has foo/fn.test5.html
// @matches - "Iter</a>&lt;'_"
pub fn test5(a: &Option<u32>) -> std::option::Iter<u32> {
    a.iter()
}

// @has foo/fn.test6.html
// @matches - "Iter</a>&lt;'_"
pub fn test6(a: &Option<u32>) -> std::option::Iter<'_, u32> {
    a.iter()
}
