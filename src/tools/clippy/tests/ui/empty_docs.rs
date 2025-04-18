//@aux-build:proc_macro_attr.rs

#![allow(unused)]
#![warn(clippy::empty_docs)]
#![allow(clippy::mixed_attributes_style)]
#![feature(extern_types)]

mod outer {
    //!
    //~^ empty_docs

    /// this is a struct
    struct Bananas {
        /// count
        count: usize,
    }

    ///
    //~^ empty_docs
    enum Warn {
        ///
        //~^ empty_docs
        A,
        B,
    }

    enum DontWarn {
        /// i
        A,
        B,
    }

    #[doc = ""]
    //~^ empty_docs
    fn warn_about_this() {}

    #[doc = ""]
    #[doc = ""]
    //~^^ empty_docs
    fn this_doesn_warn() {}

    #[doc = "a fine function"]
    fn this_is_fine() {}

    ///
    //~^ empty_docs
    mod inner {
        ///
        fn dont_warn_inner_outer() {
            //!w
        }

        fn this_is_ok() {
            //!
            //! inside the function
        }

        fn warn() {
            /*! */
            //~^ empty_docs
        }

        fn dont_warn() {
            /*! dont warn me */
        }

        trait NoDoc {
            ///
            //~^ empty_docs
            fn some() {}
        }
    }

    union Unite {
        /// lint y
        x: i32,
        ///
        //~^ empty_docs
        y: i32,
    }
}

mod issue_12377 {
    use proc_macro_attr::with_empty_docs;

    #[with_empty_docs]
    unsafe extern "C" {
        type Test;
    }

    #[with_empty_docs]
    struct Foo {
        a: u8,
    }
}
