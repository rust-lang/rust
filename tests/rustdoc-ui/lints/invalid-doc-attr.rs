#![crate_type = "lib"]
#![deny(warnings)]
#![feature(doc_masked)]

#![doc(masked)]
//~^ ERROR this attribute can only be applied to an `extern crate` item
//~| WARN is being phased out

#[doc(test(no_crate_inject))]
//~^ ERROR can only be applied at the crate level
//~| WARN is being phased out
//~| HELP to apply to the crate, use an inner attribute
//~| SUGGESTION !
#[doc(inline)]
//~^ ERROR can only be applied to a `use` item
//~| WARN is being phased out
pub fn foo() {}

pub mod bar {
    #![doc(test(no_crate_inject))]
    //~^ ERROR can only be applied at the crate level
    //~| WARN is being phased out

    #[doc(test(no_crate_inject))]
    //~^ ERROR can only be applied at the crate level
    //~| WARN is being phased out
    #[doc(inline)]
    //~^ ERROR can only be applied to a `use` item
    //~| WARN is being phased out
    pub fn baz() {}
}

#[doc(inline)]
#[doc(no_inline)]
//~^^ ERROR conflicting doc inlining attributes
//~|  HELP remove one of the conflicting attributes
pub use bar::baz;

#[doc(masked)]
//~^ ERROR this attribute can only be applied to an `extern crate` item
//~| WARN is being phased out
pub struct Masked;

#[doc(masked)]
//~^ ERROR this attribute cannot be applied to an `extern crate self` item
//~| WARN is being phased out
pub extern crate self as reexport;
