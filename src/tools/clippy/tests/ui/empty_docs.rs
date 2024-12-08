//@aux-build:proc_macro_attr.rs

#![allow(unused)]
#![warn(clippy::empty_docs)]
#![allow(clippy::mixed_attributes_style)]
#![feature(extern_types)]

mod outer {
    //!

    /// this is a struct
    struct Bananas {
        /// count
        count: usize,
    }

    ///
    enum Warn {
        ///
        A,
        B,
    }

    enum DontWarn {
        /// i
        A,
        B,
    }

    #[doc = ""]
    fn warn_about_this() {}

    #[doc = ""]
    #[doc = ""]
    fn this_doesn_warn() {}

    #[doc = "a fine function"]
    fn this_is_fine() {}

    ///
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
        }

        fn dont_warn() {
            /*! dont warn me */
        }

        trait NoDoc {
            ///
            fn some() {}
        }
    }

    union Unite {
        /// lint y
        x: i32,
        ///
        y: i32,
    }
}

mod issue_12377 {
    use proc_macro_attr::with_empty_docs;

    #[with_empty_docs]
    extern "C" {
        type Test;
    }

    #[with_empty_docs]
    struct Foo {
        a: u8,
    }
}
