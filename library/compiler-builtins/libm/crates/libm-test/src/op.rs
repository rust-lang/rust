//! Types representing individual functions.
//!
//! Each routine gets a module with its name, e.g. `mod sinf { /* ... */ }`. The module
//! contains a unit struct `Routine` which implements `MathOp`.
//!
//! Basically everything could be called a "function" here, so we loosely use the following
//! terminology:
//!
//! - "Function": the math operation that does not have an associated precision. E.g. `f(x) = e^x`,
//!   `f(x) = log(x)`.
//! - "Routine": A code implementation of a math operation with a specific precision. E.g. `exp`,
//!   `expf`, `expl`, `log`, `logf`.
//! - "Operation" / "Op": Something that relates a routine to a function or is otherwise higher
//!   level. `Op` is also used as the name for generic parameters since it is terse.

use crate::{CheckOutput, Float, TupleCall};

/// An enum representing each possible symbol name (`sin`, `sinf`, `sinl`, etc).
#[libm_macros::function_enum(BaseName)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Identifier {}

/// The name without any type specifier, e.g. `sin` and `sinf` both become `sin`.
#[libm_macros::base_name_enum]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BaseName {}

/// Attributes ascribed to a `libm` routine including signature, type information,
/// and naming.
pub trait MathOp {
    /// The float type used for this operation.
    type FTy: Float;

    /// The function type representing the signature in a C library.
    type CFn: Copy;

    /// Arguments passed to the C library function as a tuple. These may include `&mut` return
    /// values.
    type CArgs<'a>
    where
        Self: 'a;

    /// The type returned by C implementations.
    type CRet;

    /// The signature of the Rust function as a `fn(...) -> ...` type.
    type RustFn: Copy;

    /// Arguments passed to the Rust library function as a tuple.
    ///
    /// The required `TupleCall` bounds ensure this type can be passed either to the C function or
    /// to the Rust function.
    type RustArgs: Copy
        + TupleCall<Self::RustFn, Output = Self::RustRet>
        + TupleCall<Self::CFn, Output = Self::RustRet>;

    /// Type returned from the Rust function.
    type RustRet: CheckOutput<Self::RustArgs>;

    /// The name of this function, including suffix (e.g. `sin`, `sinf`).
    const IDENTIFIER: Identifier;

    /// The name as a string.
    const NAME: &'static str = Self::IDENTIFIER.as_str();

    /// The name of the function excluding the type suffix, e.g. `sin` and `sinf` are both `sin`.
    const BASE_NAME: BaseName = Self::IDENTIFIER.base_name();

    /// The function in `libm` which can be called.
    const ROUTINE: Self::RustFn;
}

macro_rules! do_thing {
    // Matcher for unary functions
    (
        fn_name: $fn_name:ident,
        FTy: $FTy:ty,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
    ) => {
        paste::paste! {
            pub mod $fn_name {
                use super::*;
                pub struct Routine;

                impl MathOp for Routine {
                    type FTy = $FTy;
                    type CFn = for<'a> $CFn;
                    type CArgs<'a> = $CArgs where Self: 'a;
                    type CRet = $CRet;
                    type RustFn = $RustFn;
                    type RustArgs = $RustArgs;
                    type RustRet = $RustRet;

                    const IDENTIFIER: Identifier = Identifier::[< $fn_name:camel >];
                    const ROUTINE: Self::RustFn = libm::$fn_name;
                }
            }

        }
    };
}

libm_macros::for_each_function! {
    callback: do_thing,
    emit_types: all,
}
