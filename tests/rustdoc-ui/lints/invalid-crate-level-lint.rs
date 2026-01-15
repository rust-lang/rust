#![crate_type = "lib"]

#[doc(test(no_crate_inject))]
//~^ ERROR can only be applied at the crate level

pub mod bar {
    #![doc(test(no_crate_inject))]
    //~^ ERROR can only be applied at the crate level

    #[doc(test(no_crate_inject))]
    //~^ ERROR can only be applied at the crate level
    fn foo() {}
}
