// edition:2018

#![feature(decl_macro)]
#![crate_name = "external_crate"]

pub mod some_module {
    /* == Make sure the logic is not affected by a re-export == */
    mod private {
        pub macro external_macro() {}
    }

    pub use private::external_macro;
}
