// https://github.com/rust-lang/rust/issues/135078
#![crate_name = "foo"]
#![feature(staged_api)]
#![stable(feature = "v1", since="1.0.0")]

#[stable(feature = "v1", since="1.0.0")]
pub mod ffi {
    #[stable(feature = "core_ffi", since="1.99.0")]
    //@ has "foo/ffi/struct.CStr.html" "//span[@class='sub-heading']/span[@class='since']" "1.99.0"
    //@ !has - "//span[@class='sub-heading']/span[@class='since']" "1.0.0"
    pub struct CStr;
}

#[stable(feature = "v1", since = "1.0.0")]
#[doc(inline)]
//@ has "foo/struct.CStr.html" "//span[@class='sub-heading']/span[@class='since']" "1.0.0"
//@ !has - "//span[@class='sub-heading']/span[@class='since']" "1.99.0"
pub use ffi::CStr;
