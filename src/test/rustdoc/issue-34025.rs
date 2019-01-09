#![crate_name = "foo"]

// @!has 'foo/sys/index.html'
// @!has 'foo/sys/sidebar-items.js'
#[doc(hidden)]
pub mod sys {
    extern "C" {
        // @!has 'foo/sys/fn.foo.html'
        #[doc(hidden)]
        pub fn foo();
    }
}
