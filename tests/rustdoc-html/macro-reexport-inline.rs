// Regression test for <https://github.com/rust-lang/rust/issues/154694>.
// The goal is to ensure that declarative macros re-exported by name
// inherit the `#[doc(inline)]` attribute from intermediate re-exports,
// matching the behavior of glob re-exports.

#![crate_name = "foo"]

#[macro_use]
mod macros {
    #[macro_export]
    #[doc(hidden)]
    macro_rules! explicit_macro {
        () => {};
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! wild_macro {
        () => {};
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! actually_hidden_macro {
        () => {};
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! actually_hidden_wild_macro {
        () => {};
    }

    #[macro_export]
    #[doc(hidden)]
    macro_rules! actually_hidden_indirect_macro {
        () => {};
    }
}

// Standard items (like structs) are provided as control cases to ensure
// macro inlining behavior maintains parity.
#[doc(hidden)]
pub struct HiddenStruct;

#[doc(hidden)]
pub struct IndirectlyHiddenStruct;

pub mod bar {
    mod hidden_explicit {
        #[doc(inline)]
        pub use crate::explicit_macro;
    }

    mod hidden_wild {
        #[doc(inline)]
        pub use crate::wild_macro;
    }

    mod actually_hidden {
        // BUG: as demonstrated by the `actually_hidden_struct` module, when both
        // `doc(hidden)` and `doc(inline)` are specified, `doc(hidden)`
        // should take priority.
        #[doc(hidden)]
        #[doc(inline)]
        pub use crate::actually_hidden_macro;
    }

    mod actually_hidden_indirect_inner {
        #[doc(inline)]
        pub use crate::actually_hidden_indirect_macro;
    }

    mod actually_hidden_indirect {
        // BUG: when there is a chain of imports, we should stop looking as soon as soon as we hit
        // something with `doc(hidden)`.
        #[doc(hidden)]
        pub use super::actually_hidden_indirect_inner::actually_hidden_indirect_macro;
    }

    mod actually_hidden_indirect_struct_inner {
        #[doc(inline)]
        pub use crate::IndirectlyHiddenStruct;
    }

    mod actually_hidden_indirect_struct {
        #[doc(hidden)]
        pub use super::actually_hidden_indirect_struct_inner::IndirectlyHiddenStruct;
    }

    mod actually_hidden_wild {
        #[doc(hidden)]
        #[doc(inline)]
        pub use crate::actually_hidden_wild_macro;
    }

    mod actually_hidden_struct {
        #[doc(inline)]
        #[doc(hidden)]
        pub use crate::HiddenStruct;
    }

    // First, we check that the explicitly named macro inherits the inline attribute
    // from `hidden_explicit` and is successfully rendered.
    //@ has 'foo/bar/macro.explicit_macro.html'
    //@ has 'foo/bar/index.html' '//a[@href="macro.explicit_macro.html"]' 'explicit_macro'
    pub use self::hidden_explicit::explicit_macro;

    // Next, we ensure that the glob-imported macro continues to render correctly
    // as a control case.
    //@ has 'foo/bar/macro.wild_macro.html'
    //@ has 'foo/bar/index.html' '//a[@href="macro.wild_macro.html"]' 'wild_macro'
    pub use self::hidden_wild::*;

    //@ !has 'foo/bar/macro.actually_hidden_macro.html'
    pub use self::actually_hidden::actually_hidden_macro;

    //@ !has 'foo/bar/macro.actually_hidden_wild_macro.html'
    pub use self::actually_hidden_wild::*;

    //@ !has 'foo/bar/struct.HiddenStruct.html'
    pub use self::actually_hidden_struct::HiddenStruct;

    //@ !has 'foo/bar/macro.actually_hidden_indirect_macro.html'
    pub use self::actually_hidden_indirect::actually_hidden_indirect_macro;

    //@ !has 'foo/bar/struct.IndirectlyHiddenStruct.html'
    pub use self::actually_hidden_indirect_struct::IndirectlyHiddenStruct;
}
