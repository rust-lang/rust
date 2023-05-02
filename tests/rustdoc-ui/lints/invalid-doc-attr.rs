#![crate_type = "lib"]
#![deny(warnings)]

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
