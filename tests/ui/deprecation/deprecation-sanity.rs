#![deny(unused_attributes)]

// Various checks that deprecation attributes are used correctly

mod bogus_attribute_types_1 {
    #[deprecated(since = "a", note = "a", reason)] //~ ERROR unknown meta item 'reason'
    fn f1() { }

    #[deprecated(since = "a", note)] //~ ERROR malformed `deprecated` attribute input [E0539]
    fn f2() { }

    #[deprecated(since, note = "a")] //~ ERROR malformed `deprecated` attribute input [E0539]
    fn f3() { }

    #[deprecated(since = "a", note(b))] //~ ERROR malformed `deprecated` attribute input [E0539]
    fn f5() { }

    #[deprecated(since(b), note = "a")] //~ ERROR malformed `deprecated` attribute input [E0539]
    fn f6() { }

    #[deprecated(note = b"test")] //~ ERROR malformed `deprecated` attribute input [E0539]
    fn f7() { }

    #[deprecated("test")] //~ ERROR malformed `deprecated` attribute input [E0565]
    fn f8() { }
}

#[deprecated(since = "a", note = "b")]
#[deprecated(since = "a", note = "b")] //~ ERROR multiple `deprecated` attributes
fn multiple1() { }

#[deprecated(since = "a", since = "b", note = "c")] //~ ERROR malformed `deprecated` attribute input [E0538]
fn f1() { }

struct X;

#[deprecated = "hello"] //~ ERROR attribute cannot be used on
//~| WARN previously accepted
impl Default for X {
    fn default() -> Self {
        X
    }
}

unsafe extern "C" {
    #[deprecated]
    static FOO: std::ffi::c_int;
}

fn main() { }
