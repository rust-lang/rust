#![crate_name = "issue162605"]

//@ count 'issue162605/foo/index.html' '//*[@id="reexport.inlined_macro"]' 1
//@ has 'issue162605/bar/macro.inlined_macro.html'
//@ !has 'issue162605/foo/macro.inlined_macro.html'
//@ !has 'issue162605/macro.hidden_macro.html'

pub mod foo {
    // inlined_macro does not appear in foo module.
    #[doc(no_inline)]
    pub use super::bar::inlined_macro;
}

pub mod bar {
    // inlined_macro appears in bar module.
    #[doc(inline)]
    pub use crate::hidden_macro as inlined_macro;

    #[macro_export]
    #[doc(hidden)]
    macro_rules! hidden_macro {
        () => {
            "dummy"
        };
    }
}
