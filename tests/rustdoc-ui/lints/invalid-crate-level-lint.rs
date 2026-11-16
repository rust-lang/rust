#![deny(invalid_doc_attributes)]
#![crate_type = "lib"]

#[doc(test(no_crate_inject))]
//~^ ERROR can only be applied at the crate level

pub mod bar {
    #![doc(test(no_crate_inject))]
    //~^ ERROR unused attribute
    //~| WARN this was previously accepted by the compiler but is being phased out

    #[doc(test(no_crate_inject))]
    //~^ ERROR can only be applied at the crate level
    fn foo() {}
}
