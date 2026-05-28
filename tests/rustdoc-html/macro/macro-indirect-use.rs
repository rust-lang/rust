// Checks that it is possible to make a macro public through a `pub use` of its
// parent module.
//
// This is a regression test for issue #87257.

#![feature(decl_macro)]

mod outer {
    pub mod inner {
        pub macro some_macro() {}
    }
}

//@ has macro_indirect_use/inner/index.html
//@ has macro_indirect_use/inner/macro.some_macro.html
pub use outer::inner;
