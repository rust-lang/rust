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
//! `rustc_mir_build/src/build/custom/mod.rs`.
//!
//! Typical usage will look like this:
//!
//! ```rust
//! #![feature(core_intrinsics, custom_mir)]
//! #![allow(internal_features)]
//!
//! use core::intrinsics::mir::*;
//!
//! #[custom_mir(dialect = "built")]
//! pub fn simple(x: i32) -> i32 {
//!     mir! {
//!         let temp2: i32;
//!
//!         {
//!             let temp1 = x;
//!             Goto(my_second_block)
//!         }
//!
//!         my_second_block = {
//!             temp2 = Move(temp1);
//!             RET = temp2;
//!             Return()
//!         }
//!     }
//! }
//! ```
//!
//! The `custom_mir` attribute tells the compiler to treat the function as being custom MIR. This
//! attribute only works on functions - there is no way to insert custom MIR into the middle of
//! another function. The `dialect` and `phase` parameters indicate which [version of MIR][dialect
//! docs] you are inserting here. Generally you'll want to use `#![custom_mir(dialect = "built")]`
//! if you want your MIR to be modified by the full MIR pipeline, or `#![custom_mir(dialect =
//! "runtime", phase = "optimized")]` if you don't.
//!
//! [dialect docs]:
//!     https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/mir/enum.MirPhase.html
//!
//! The input to the [`mir!`] macro is:
//!
//!  - An optional return type annotation in the form of `type RET = ...;`. This may be required
//!    if the compiler cannot infer the type of RET.
//!  - A possibly empty list of local declarations. Locals can also be declared inline on
//!    assignments via `let`. Type inference generally works. Shadowing does not.
//!  - A list of basic blocks. The first of these is the start block and is where execution begins.
//!    All blocks other than the start block need to be given a name, so that they can be referred
//!    to later.
//!     - Each block is a list of semicolon terminated statements, followed by a terminator. The
//!       syntax for the various statements and terminators is designed to be as similar as possible
//!       to the syntax for analogous concepts in native Rust. See below for a list.
//!
//! # Examples
//!
//! ```rust
//! #![feature(core_intrinsics, custom_mir)]
//! #![allow(internal_features)]
//! #![allow(unused_assignments)]
//!
//! use core::intrinsics::mir::*;
//!
//! #[custom_mir(dialect = "built")]
//! pub fn choose_load(a: &i32, b: &i32, c: bool) -> i32 {
//!     mir! {
//!         {
//!             match c {
//!                 true => t,
//!                 _ => f,
//!             }
//!         }
//!
//!         t = {
//!             let temp = a;
//!             Goto(load_and_exit)
//!         }
//!
//!         f = {
//!             temp = b;
//!             Goto(load_and_exit)
//!         }
//!
//!         load_and_exit = {
//!             RET = *temp;
//!             Return()
//!         }
//!     }
//! }
//!
//! #[custom_mir(dialect = "built")]
//! fn unwrap_unchecked<T>(opt: Option<T>) -> T {
//!     mir! {
//!         {
//!             RET = Move(Field(Variant(opt, 1), 0));
//!             Return()
//!         }
//!     }
//! }
//!
//! #[custom_mir(dialect = "runtime", phase = "optimized")]
//! fn push_and_pop<T>(v: &mut Vec<T>, value: T) {
//!     mir! {
//!         let _unused;
//!         let popped;
//!
//!         {
//!             Call(_unused = Vec::push(v, value), ReturnTo(pop), UnwindContinue())
//!         }
//!
//!         pop = {
//!             Call(popped = Vec::pop(v), ReturnTo(drop), UnwindContinue())
//!         }
//!
//!         drop = {
//!             Drop(popped, ReturnTo(ret), UnwindContinue())
//!         }
//!
//!         ret = {
//!             Return()
//!         }
//!     }
//! }
//!
//! #[custom_mir(dialect = "runtime", phase = "optimized")]
//! fn annotated_return_type() -> (i32, bool) {
//!     mir! {
//!         type RET = (i32, bool);
//!         {
//!             RET.0 = 1;
//!             RET.1 = true;
//!             Return()
//!         }
//!     }
//! }
//! ```
//!
//! We can also set off compilation failures that happen in sufficiently late stages of the
//! compiler:
//!
//! ```rust,compile_fail
//! #![feature(core_intrinsics, custom_mir)]
//!
//! extern crate core;
//! use core::intrinsics::mir::*;
//!
//! #[custom_mir(dialect = "built")]
//! fn borrow_error(should_init: bool) -> i32 {
//!     mir! {
//!         let temp: i32;
//!
//!         {
//!             match should_init {
//!                 true => init,
//!                 _ => use_temp,
//!             }
//!         }
//!
//!         init = {
//!             temp = 0;
//!             Goto(use_temp)
//!         }
//!
//!         use_temp = {
//!             RET = temp;
//!             Return()
//!         }
//!     }
//! }
//! ```
//!
//! ```text
//! error[E0381]: used binding is possibly-uninitialized
//!   --> test.rs:24:13
//!    |
//! 8  | /     mir! {
//! 9  | |         let temp: i32;
//! 10 | |
//! 11 | |         {
//! ...  |
//! 19 | |             temp = 0;
//!    | |             -------- binding initialized here in some conditions
//! ...  |
//! 24 | |             RET = temp;
//!    | |             ^^^^^^^^^^ value used here but it is possibly-uninitialized
//! 25 | |             Return()
//! 26 | |         }
//! 27 | |     }
//!    | |_____- binding declared here but left uninitialized
//!
//! error: aborting due to 1 previous error
//!
//! For more information about this error, try `rustc --explain E0381`.
//! ```
//!
//! # Syntax
//!
//! The lists below are an exhaustive description of how various MIR constructs can be created.
//! Anything missing from the list should be assumed to not be supported, PRs welcome.
//!
//! #### Locals
//!
//!  - The `_0` return local can always be accessed via `RET`.
//!  - Arguments can be accessed via their regular name.
//!  - All other locals need to be declared with `let` somewhere and then can be accessed by name.
//!
//! #### Places
//!  - Locals implicitly convert to places.
//!  - Field accesses, derefs, and indexing work normally.
//!  - Fields in variants can be accessed via the [`Variant`] and [`Field`] associated functions,
//!    see their documentation for details.
//!
//! #### Operands
//!  - Places implicitly convert to `Copy` operands.
//!  - `Move` operands can be created via [`Move`].
//!  - Const blocks, literals, named constants, and const params all just work.
//!  - [`Static`] and [`StaticMut`] can be used to create `&T` and `*mut T`s to statics. These are
//!    constants in MIR and the only way to access statics.
//!
//! #### Statements
//!  - Assign statements work via normal Rust assignment.
//!  - [`Retag`], [`StorageLive`], [`StorageDead`], [`Deinit`] statements have an associated function.
//!
//! #### Rvalues
//!
//!  - Operands implicitly convert to `Use` rvalues.
//!  - `&`, `&mut`, `addr_of!`, and `addr_of_mut!` all work to create their associated rvalue.
//!  - [`CopyForDeref`], [`CastTransmute`], [`CastPtrToPtr`], [`CastUnsize`], and [`Discriminant`]
//!    have associated functions.
//!  - Unary and binary operations use their normal Rust syntax - `a * b`, `!c`, etc.
//!  - The binary operation `Offset` can be created via [`Offset`].
//!  - Checked binary operations are represented by wrapping the associated binop in [`Checked`].
//!  - Array repetition syntax (`[foo; 10]`) creates the associated rvalue.
//!
//! #### Terminators
//!
//!  - [`Goto`], [`Return`], [`Unreachable`] and [`Drop`](Drop()) have associated functions.
//!  - `match some_int_operand` becomes a `SwitchInt`. Each arm should be `literal => basic_block`
//!     - The exception is the last arm, which must be `_ => basic_block` and corresponds to the
//!       otherwise branch.
//!  - [`Call`] has an associated function as well, with special syntax:
//!    `Call(ret_val = function(arg1, arg2, ...), ReturnTo(next_block), UnwindContinue())`.
//!  - [`TailCall`] does not have a return destination or next block, so its syntax is just
//!    `TailCall(function(arg1, arg2, ...))`.
//!
//! #### Debuginfo
//!
//! Debuginfo associates source code variable names (of variables that may not exist any more) with
//! MIR expressions that indicate where the value of that variable is stored. The syntax to do so
//! is:
//! ```text
//! debug source_var_name => expression;
//! ```
//! Both places and constants are supported in the `expression`.
//!
//! ```rust
//! #![allow(internal_features)]
//! #![feature(core_intrinsics, custom_mir)]
//!
//! use core::intrinsics::mir::*;
//!
//! #[custom_mir(dialect = "built")]
//! fn debuginfo(arg: Option<&i32>) {
//!     mir!(
//!         // Debuginfo for a source variable `plain_local` that just duplicates `arg`.
//!         debug plain_local => arg;
//!         // Debuginfo for a source variable `projection` that can be computed by dereferencing
//!         // a field of `arg`.
//!         debug projection => *Field::<&i32>(Variant(arg, 1), 0);
//!         // Debuginfo for a source variable `constant` that always holds the value `5`.
//!         debug constant => 5_usize;
//!         {
//!             Return()
//!         }
//!     )
//! }
//! ```

#![unstable(
    feature = "custom_mir",
    reason = "MIR is an implementation detail and extremely unstable",
    issue = "none"
)]
#![allow(unused_variables, non_snake_case, missing_debug_implementations)]

/// Type representing basic blocks.
///
/// All terminators will have this type as a return type. It helps achieve some type safety.
#[rustc_diagnostic_item = "mir_basic_block"]
pub enum BasicBlock {
    /// A non-cleanup basic block.
    Normal,
    /// A basic block that lies on an unwind path.
    Cleanup,
}

/// The reason we are terminating the process during unwinding.
#[rustc_diagnostic_item = "mir_unwind_terminate_reason"]
pub enum UnwindTerminateReason {
    /// Unwinding is just not possible given the ABI of this function.
    Abi,
    /// We were already cleaning up for an ongoing unwind, and a *second*, *nested* unwind was
    /// triggered by the drop glue.
    InCleanup,
}

pub use UnwindTerminateReason::{Abi as ReasonAbi, InCleanup as ReasonInCleanup};

macro_rules! define {
    ($name:literal, $( #[ $meta:meta ] )* fn $($sig:tt)*) => {
        #[rustc_diagnostic_item = $name]
        #[inline]
        $( #[ $meta ] )*
        pub fn $($sig)* { panic!() }
    }
}

// Unwind actions
pub struct UnwindActionArg;
define!(
    "mir_unwind_continue",
    /// An unwind action that continues unwinding.
    fn UnwindContinue() -> UnwindActionArg
);
define!(
    "mir_unwind_unreachable",
    /// An unwind action that triggers undefined behavior.
    fn UnwindUnreachable() -> UnwindActionArg
);
define!(
    "mir_unwind_terminate",
    /// An unwind action that terminates the execution.
    ///
    /// `UnwindTerminate` can also be used as a terminator.
    fn UnwindTerminate(reason: UnwindTerminateReason) -> UnwindActionArg
);
define!(
    "mir_unwind_cleanup",
    /// An unwind action that continues execution in a given basic block.
    fn UnwindCleanup(goto: BasicBlock) -> UnwindActionArg
);

// Return destination for `Call`
pub struct ReturnToArg;
define!("mir_return_to", fn ReturnTo(goto: BasicBlock) -> ReturnToArg);

// Terminators
define!("mir_return", fn Return() -> BasicBlock);
define!("mir_goto", fn Goto(destination: BasicBlock) -> BasicBlock);
define!("mir_unreachable", fn Unreachable() -> BasicBlock);
define!("mir_drop",
    /// Drop the contents of a place.
    ///
    /// The first argument must be a place.
    ///
    /// The second argument must be of the form `ReturnTo(bb)`, where `bb` is the basic block that
    /// will be jumped to after the destructor returns.
    ///
    /// The third argument describes what happens on unwind. It can be one of:
    /// - [`UnwindContinue`]
    /// - [`UnwindUnreachable`]
    /// - [`UnwindTerminate`]
    /// - [`UnwindCleanup`]
    fn Drop<T>(place: T, goto: ReturnToArg, unwind_action: UnwindActionArg)
);
define!("mir_call",
    /// Call a function.
    ///
    /// The first argument must be of the form `ret_val = fun(arg1, arg2, ...)`.
    ///
    /// The second argument must be of the form `ReturnTo(bb)`, where `bb` is the basic block that
    /// will be jumped to after the function returns.
    ///
    /// The third argument describes what happens on unwind. It can be one of:
    /// - [`UnwindContinue`]
    /// - [`UnwindUnreachable`]
    /// - [`UnwindTerminate`]
    /// - [`UnwindCleanup`]
    fn Call(call: (), goto: ReturnToArg, unwind_action: UnwindActionArg)
);
define!("mir_tail_call",
    /// Call a function.
    ///
    /// The argument must be of the form `fun(arg1, arg2, ...)`.
    fn TailCall<T>(call: T)
);
define!("mir_unwind_resume",
    /// A terminator that resumes the unwinding.
    fn UnwindResume()
);

define!("mir_storage_live", fn StorageLive<T>(local: T));
define!("mir_storage_dead", fn StorageDead<T>(local: T));
define!("mir_assume", fn Assume(operand: bool));
define!("mir_deinit", fn Deinit<T>(place: T));
define!("mir_checked", fn Checked<T>(binop: T) -> (T, bool));
define!(
    "mir_ptr_metadata",
    fn PtrMetadata<P: ?Sized>(place: *const P) -> <P as ::core::ptr::Pointee>::Metadata
);
define!("mir_copy_for_deref", fn CopyForDeref<T>(place: T) -> T);
define!("mir_retag", fn Retag<T>(place: T));
define!("mir_move", fn Move<T>(place: T) -> T);
define!("mir_static", fn Static<T>(s: T) -> &'static T);
define!("mir_static_mut", fn StaticMut<T>(s: T) -> *mut T);
define!(
    "mir_discriminant",
    /// Gets the discriminant of a place.
    fn Discriminant<T>(place: T) -> <T as ::core::marker::DiscriminantKind>::Discriminant
);
define!("mir_set_discriminant", fn SetDiscriminant<T>(place: T, index: u32));
define!("mir_offset", fn Offset<T, U>(ptr: T, count: U) -> T);
define!(
    "mir_field",
    /// Access the field with the given index of some place.
    ///
    /// This only makes sense to use in conjunction with [`Variant`]. If the type you are looking to
    /// access the field of does not have variants, you can use normal field projection syntax.
    ///
    /// There is no proper way to do a place projection to a variant in Rust, and so these two
    /// functions are a workaround. You can access a field of a variant via `Field(Variant(place,
    /// var_idx), field_idx)`, where `var_idx` and `field_idx` are appropriate literals. Some
    /// caveats:
    ///
    ///  - The return type of `Variant` is always `()`. Don't worry about that, the correct MIR will
    ///    still be generated.
    ///  - In some situations, the return type of `Field` cannot be inferred. You may need to
    ///    annotate it on the function in these cases.
    ///  - Since `Field` is a function call which is not a place expression, using this on the left
    ///    hand side of an expression is rejected by the compiler. [`place!`] is a macro provided to
    ///    work around that issue. Wrap the left hand side of an assignment in the macro to convince
    ///    the compiler that it's ok.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![allow(internal_features)]
    /// #![feature(custom_mir, core_intrinsics)]
    ///
    /// use core::intrinsics::mir::*;
    ///
    /// #[custom_mir(dialect = "built")]
    /// fn unwrap_deref(opt: Option<&i32>) -> i32 {
    ///     mir! {
    ///         {
    ///             RET = *Field::<&i32>(Variant(opt, 1), 0);
    ///             Return()
    ///         }
    ///     }
    /// }
    ///
    /// #[custom_mir(dialect = "built")]
    /// fn set(opt: &mut Option<i32>) {
    ///     mir! {
    ///         {
    ///             place!(Field(Variant(*opt, 1), 0)) = 5;
    ///             Return()
    ///         }
    ///     }
    /// }
    /// ```
    fn Field<F>(place: (), field: u32) -> F
);
define!(
    "mir_variant",
    /// Adds a variant projection with the given index to the place.
    ///
    /// See [`Field`] for documentation.
    fn Variant<T>(place: T, index: u32) -> ()
);
define!(
    "mir_cast_transmute",
    /// Emits a `CastKind::Transmute` cast.
    ///
    /// Needed to test the UB when `sizeof(T) != sizeof(U)`, which can't be
    /// generated via the normal `mem::transmute`.
    fn CastTransmute<T, U>(operand: T) -> U
);
define!(
    "mir_cast_ptr_to_ptr",
    /// Emits a `CastKind::PtrToPtr` cast.
    ///
    /// This allows bypassing normal validation to generate strange casts.
    fn CastPtrToPtr<T, U>(operand: T) -> U
);
define!(
    "mir_cast_unsize",
    /// Emits a `CastKind::PointerCoercion(Unsize)` cast.
    ///
    /// This allows bypassing normal validation to generate strange casts.
    fn CastUnsize<T, U>(operand: T) -> U
);
define!(
    "mir_make_place",
    #[doc(hidden)]
    fn __internal_make_place<T>(place: T) -> *mut T
);
define!(
    "mir_debuginfo",
    #[doc(hidden)]
    fn __debuginfo<T>(name: &'static str, s: T)
);

/// Macro for generating custom MIR.
///
/// See the module documentation for syntax details. This macro is not magic - it only transforms
/// your MIR into something that is easier to parse in the compiler.
#[rustc_macro_transparency = "transparent"]
pub macro mir {
    {
        $(type RET = $ret_ty:ty ;)?
        $(let $local_decl:ident $(: $local_decl_ty:ty)? ;)*
        $(debug $dbg_name:ident => $dbg_data:expr ;)*

        {
            $($entry:tt)*
        }

        $(
            $block_name:ident $(($block_cleanup:ident))? = {
                $($block:tt)*
            }
        )*
    } => {{
        // First, we declare all basic blocks.
        __internal_declare_basic_blocks!($(
            $block_name $(($block_cleanup))?
        )*);
        {
            // Now all locals
            #[allow(non_snake_case)]
            let RET $(: $ret_ty)?;
            $(
                let $local_decl $(: $local_decl_ty)? ;
            )*
            ::core::intrinsics::mir::__internal_extract_let!($($entry)*);
            $(
                ::core::intrinsics::mir::__internal_extract_let!($($block)*);
            )*

            {
                // Now debuginfo
                $(
                    __debuginfo(stringify!($dbg_name), $dbg_data);
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
        }
    }}
}

/// Helper macro that allows you to treat a value expression like a place expression.
///
/// See the documentation on [`Variant`] for why this is necessary and how to use it.
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

/// Helper macro that declares the basic blocks.
#[doc(hidden)]
pub macro __internal_declare_basic_blocks {
    () => {},
    ($name:ident (cleanup) $($rest:tt)*) => {
        let $name = ::core::intrinsics::mir::BasicBlock::Cleanup;
        __internal_declare_basic_blocks!($($rest)*)
    },
    ($name:ident $($rest:tt)*) => {
        let $name = ::core::intrinsics::mir::BasicBlock::Normal;
        __internal_declare_basic_blocks!($($rest)*)
    },
}
