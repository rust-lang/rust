//@ check-pass

macro_rules! m {
    () => {
        /// A
        //~^ WARNING
        #[path = "auxiliary/module_macro_doc.rs"]
        pub mod mymodule;
    }
}

m!();
