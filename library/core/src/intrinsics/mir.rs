//! Rustc internal tooling for hand-writing MIR.
//!
//! If for some reasons you are not writing rustc tests and have found yourself considering using
//! this feature, turn back. This is *exceptionally* unstable. There is no attempt at all to make
//! anything work besides those things which the rustc test suite happened to need. If you make a
//! typo you'll probably ICE. Really, this is not the solution to your problems. Consider instead
//! supporting the [stable MIR project group](https://github.com/rust-lang/project-stable-mir).
//!
//! The documentation for this module describes how to use this feature. If you are interested in
//! hacking on the implementation, most of that documentation lives at
//! `rustc_mir_building/src/build/custom/mod.rs`.
//!
//! Typical usage will look like this:
//!
//! ```rust
//! #![feature(core_intrinsics, custom_mir)]
//!
//! extern crate core;
//! use core::intrinsics::mir::*;
//!
//! #[custom_mir(dialect = "built")]
//! pub fn simple(x: i32) -> i32 {
//!     mir!(
//!         let temp1: i32;
//!         let temp2: _;
//!
//!         {
//!             temp1 = x;
//!             Goto(exit)
//!         }
//!
//!         exit = {
//!             temp2 = Move(temp1);
//!             RET = temp2;
//!             Return()
//!         }
//!     )
//! }
//! ```
//!
//! Hopefully most of this is fairly self-explanatory. Expanding on some notable details:
//!
//!  - The `custom_mir` attribute tells the compiler to treat the function as being custom MIR. This
//!    attribute only works on functions - there is no way to insert custom MIR into the middle of
//!    another function.
//!  - The `dialect` and `phase` parameters indicate which version of MIR you are inserting here.
//!    This will normally be the phase that corresponds to the thing you are trying to test. The
//!    phase can be omitted for dialects that have just one.
//!  - You should define your function signature like you normally would. Externally, this function
//!    can be called like any other function.
//!  - Type inference works - you don't have to spell out the type of all of your locals.
//!
//! For now, all statements and terminators are parsed from nested invocations of the special
//! functions provided in this module. We additionally want to (but do not yet) support more
//! "normal" Rust syntax in places where it makes sense. Also, most kinds of instructions are not
//! supported yet.
//!

#![unstable(
    feature = "custom_mir",
    reason = "MIR is an implementation detail and extremely unstable",
    issue = "none"
)]
#![allow(unused_variables, non_snake_case, missing_debug_implementations)]

/// Type representing basic blocks.
///
/// All terminators will have this type as a return type. It helps achieve some type safety.
pub struct BasicBlock;

macro_rules! define {
    ($name:literal, $($sig:tt)*) => {
        #[rustc_diagnostic_item = $name]
        pub $($sig)* { panic!() }
    }
}

define!("mir_return", fn Return() -> BasicBlock);
define!("mir_goto", fn Goto(destination: BasicBlock) -> BasicBlock);
define!("mir_retag", fn Retag<T>(place: T));
define!("mir_retag_raw", fn RetagRaw<T>(place: T));
define!("mir_move", fn Move<T>(place: T) -> T);
define!("mir_static", fn Static<T>(s: T) -> &'static T);
define!("mir_static_mut", fn StaticMut<T>(s: T) -> *mut T);
define!(
    "mir_discriminant",
    /// Gets the discriminant of a place.
    fn Discriminant<T>(place: T) -> <T as ::core::marker::DiscriminantKind>::Discriminant
);
define!("mir_set_discriminant", fn SetDiscriminant<T>(place: T, index: u32));
define!("mir_field", fn Field<F>(place: (), field: u32) -> F);
define!("mir_variant", fn Variant<T>(place: T, index: u32) -> ());
define!("mir_make_place", fn __internal_make_place<T>(place: T) -> *mut T);

/// Convenience macro for generating custom MIR.
///
/// See the module documentation for syntax details. This macro is not magic - it only transforms
/// your MIR into something that is easier to parse in the compiler.
#[rustc_macro_transparency = "transparent"]
pub macro mir {
    (
        $(let $local_decl:ident $(: $local_decl_ty:ty)? ;)*

        {
            $($entry:tt)*
        }

        $(
            $block_name:ident = {
                $($block:tt)*
            }
        )*
    ) => {{
        // First, we declare all basic blocks.
        $(
            let $block_name: ::core::intrinsics::mir::BasicBlock;
        )*

        {
            // Now all locals
            #[allow(non_snake_case)]
            let RET;
            $(
                let $local_decl $(: $local_decl_ty)? ;
            )*

            ::core::intrinsics::mir::__internal_extract_let!($($entry)*);
            $(
                ::core::intrinsics::mir::__internal_extract_let!($($block)*);
            )*

            {
                // Finally, the contents of the basic blocks
                ::core::intrinsics::mir::__internal_remove_let!({
                    {}
                    { $($entry)* }
                });
                $(
                    ::core::intrinsics::mir::__internal_remove_let!({
                        {}
                        { $($block)* }
                    });
                )*

                RET
            }
        }
    }}
}

/// Helper macro that allows you to treat a value expression like a place expression.
///
/// This is necessary in combination with the [`Field`] and [`Variant`] methods. Specifically,
/// something like this won't compile on its own, reporting an error about not being able to assign
/// to such an expression:
///
/// ```rust,ignore(syntax-highlighting-only)
/// Field(something, 0) = 5;
/// ```
///
/// Instead, you'll need to write
///
/// ```rust,ignore(syntax-highlighting-only)
/// place!(Field(something, 0)) = 5;
/// ```
pub macro place($e:expr) {
    (*::core::intrinsics::mir::__internal_make_place($e))
}

/// Helper macro that extracts the `let` declarations out of a bunch of statements.
///
/// This macro is written using the "statement muncher" strategy. Each invocation parses the first
/// statement out of the input, does the appropriate thing with it, and then recursively calls the
/// same macro on the remainder of the input.
#[doc(hidden)]
pub macro __internal_extract_let {
    // If it's a `let` like statement, keep the `let`
    (
        let $var:ident $(: $ty:ty)? = $expr:expr; $($rest:tt)*
    ) => {
        let $var $(: $ty)?;
        ::core::intrinsics::mir::__internal_extract_let!($($rest)*);
    },
    // Due to #86730, we have to handle const blocks separately
    (
        let $var:ident $(: $ty:ty)? = const $block:block; $($rest:tt)*
    ) => {
        let $var $(: $ty)?;
        ::core::intrinsics::mir::__internal_extract_let!($($rest)*);
    },
    // Otherwise, output nothing
    (
        $stmt:stmt; $($rest:tt)*
    ) => {
        ::core::intrinsics::mir::__internal_extract_let!($($rest)*);
    },
    (
        $expr:expr
    ) => {}
}

/// Helper macro that removes the `let` declarations from a bunch of statements.
///
/// Because expression position macros cannot expand to statements + expressions, we need to be
/// slightly creative here. The general strategy is also statement munching as above, but the output
/// of the macro is "stored" in the subsequent macro invocation. Easiest understood via example:
/// ```text
/// invoke!(
///     {
///         {
///             x = 5;
///         }
///         {
///             let d = e;
///             Call()
///         }
///     }
/// )
/// ```
/// becomes
/// ```text
/// invoke!(
///     {
///         {
///             x = 5;
///             d = e;
///         }
///         {
///             Call()
///         }
///     }
/// )
/// ```
#[doc(hidden)]
pub macro __internal_remove_let {
    // If it's a `let` like statement, remove the `let`
    (
        {
            {
                $($already_parsed:tt)*
            }
            {
                let $var:ident $(: $ty:ty)? = $expr:expr;
                $($rest:tt)*
            }
        }
    ) => { ::core::intrinsics::mir::__internal_remove_let!(
        {
            {
                $($already_parsed)*
                $var = $expr;
            }
            {
                $($rest)*
            }
        }
    )},
    // Due to #86730 , we have to handle const blocks separately
    (
        {
            {
                $($already_parsed:tt)*
            }
            {
                let $var:ident $(: $ty:ty)? = const $block:block;
                $($rest:tt)*
            }
        }
    ) => { ::core::intrinsics::mir::__internal_remove_let!(
        {
            {
                $($already_parsed)*
                $var = const $block;
            }
            {
                $($rest)*
            }
        }
    )},
    // Otherwise, keep going
    (
        {
            {
                $($already_parsed:tt)*
            }
            {
                $stmt:stmt;
                $($rest:tt)*
            }
        }
    ) => { ::core::intrinsics::mir::__internal_remove_let!(
        {
            {
                $($already_parsed)*
                $stmt;
            }
            {
                $($rest)*
            }
        }
    )},
    (
        {
            {
                $($already_parsed:tt)*
            }
            {
                $expr:expr
            }
        }
    ) => {
        {
            $($already_parsed)*
            $expr
        }
    },
}
