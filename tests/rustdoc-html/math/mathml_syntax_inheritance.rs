// This test ensures that the mathml.css file is loaded if,
// and only if, there is a math span in the file.

#![feature(rustdoc_texmath)]
#![crate_name = "foo"]
#![doc(syntax="-tex_math_dollars")]

//! disabled at crate level
//! 
//! $\sqrt{2}$
//@ has 'foo/index.html'
//@ count - '//math' 0

#[doc(syntax="+tex_math_dollars")]
/// enabled in this module
/// 
/// $\sqrt{2}$
//@ has 'foo/a/index.html'
//@ count - '//math' 1
pub mod a {}

#[doc(syntax="-tex_math_dollars")]
/// disabled in this module $\sqrt{2}$
//@ has 'foo/b/index.html'
//@ count - '//math' 0
pub mod b {
    #[doc(syntax="+tex_math_dollars")]
    /// enabled on this function
    /// 
    /// $\sqrt{2}$
    //@ has 'foo/b/fn.foo.html'
    //@ count - '//math' 1
    pub fn foo() {}
}

/// disabled on this struct $\sqrt{2}$
//@ has 'foo/struct.C.html'
//@ count - '//math' 1
pub struct C {
    #[doc(syntax="+tex_math_dollars")]
    /// enabled on this field $\sqrt{2}$
    pub field: bool,
}

/// disabled on this struct $\sqrt{2}$
//@ has 'foo/struct.D.html'
//@ count - '//math' 2
pub struct D;

#[doc(syntax="+tex_math_dollars")]
/// enabled on this impl $\sqrt{2}$
impl D {
    /// enabled on this impl $\sqrt{2}$
    pub fn foo() {}
}

/// disabled on this struct $\sqrt{2}$
//@ has 'foo/struct.E.html'
//@ count - '//math' 1
pub struct E;

/// disabled on this impl $\sqrt{2}$
impl E {
    #[doc(syntax="+tex_math_dollars")]
    /// enabled on this fn $\sqrt{2}$
    pub fn foo() {}
}

/// enabled on this struct
//@ has 'foo/struct.F.html'
//@ count - '//math' 0
#[doc(syntax="+tex_math_dollars")]
pub struct F;

/// disabled on this impl $\sqrt{2}$
impl F {
    /// disabled on this fn $\sqrt{2}$
    pub fn foo() {}
}

unsafe extern "C" {
    /// disabled on this fn
    ///
    /// $\sqrt{2}$
    //@ has 'foo/fn.g.html'
    //@ count - '//math' 0
    pub unsafe fn g();
    /// enabled on this fn
    ///
    /// $\sqrt{2}$
    //@ has 'foo/fn.h.html'
    //@ count - '//math' 1
    #[doc(syntax="+tex_math_dollars")]
    pub unsafe fn h();
}

#[allow(unused_doc_comments)]
#[doc(syntax="+tex_math_dollars")]
unsafe extern "C" {
    /// doc attributes do not work on extern blocks
    ///
    /// $\sqrt{2}$
    //@ has 'foo/fn.i.html'
    //@ count - '//math' 0
    pub unsafe fn i();
}

/// disabled on this mod, but enabled on some children $\sqrt{2}$
//@ has 'foo/j/index.html'
//@ count - '//math' 1
pub mod j {
    /// $\sqrt{2}$
    //@ has 'foo/j/fn.j_1.html'
    //@ count - '//math' 0
    pub fn j_1() {}
    /// $\sqrt{2}$
    //@ has 'foo/j/fn.j_2.html'
    //@ count - '//math' 1
    #[doc(syntax="+tex_math_dollars")]
    pub fn j_2() {}
    /// foo
    /// 
    /// $\sqrt{2}$
    //@ has 'foo/j/fn.j_3.html'
    //@ count - '//math' 1
    #[doc(syntax="+tex_math_dollars")]
    pub fn j_3() {}
}
