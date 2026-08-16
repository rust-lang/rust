#![feature(rustdoc_texmath)]
#![deny(unused_doc_comments)]

#[doc(syntax(enable(tex_math_dollars)))]
pub fn foo() {}

pub fn bar() {
    #[doc(syntax(enable(tex_math_dollars)))]
    // there should be an unused_doc_comments line here
    fn baz() {}
    #[doc(syntax(enable(tex_math_dollars)))]
    // there should be an unused_doc_comments line here
    struct Baz;
}

#[doc(syntax(enable(tex_math_dollars)))]
//~^ ERROR unused_doc_comments
unsafe extern "C" {
    pub unsafe fn quux();
}

unsafe extern "C" {
    #[doc(syntax(enable(tex_math_dollars)))]
    pub unsafe fn zzyzx();
}
