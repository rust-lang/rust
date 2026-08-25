// `S::<S2>::foo` and `S::<S1>::foo` have same `DefPath` modulo disambiguator.
// `libr.rs` interface may not contain `S::<S1>::foo` as private items aren't
// exportable. We should make sure that original `S::<S2>::foo` and the one
// produced during interface generation have same mangled names.

#![feature(export_stable)]
#![crate_type = "sdylib"]

#[export_stable]
#[repr(C)]
pub struct S<T>(pub T);

struct S1;
pub struct S2;

impl S<S1> {
    extern "C" fn foo() -> i32 {
        1
    }
}

#[export_stable]
impl S<S2> {
    pub extern "C" fn foo() -> i32 {
        2
    }
}
