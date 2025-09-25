mod crosspointer_transmute;
mod eager_transmute;
mod missing_transmute_annotations;
mod transmute_int_to_bool;
mod transmute_int_to_non_zero;
mod transmute_null_to_fn;
mod transmute_ptr_to_ptr;
mod transmute_ptr_to_ref;
mod transmute_ref_to_ref;
mod transmute_undefined_repr;
mod transmutes_expressible_as_ptr_casts;
mod transmuting_null;
mod unsound_collection_transmute;
mod useless_transmute;
mod utils;
mod wrong_transmute;

use clippy_config::Conf;
use clippy_utils::is_in_const_context;
use clippy_utils::msrvs::Msrv;
use rustc_hir::{Expr, ExprKind, QPath};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::impl_lint_pass;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes that can't ever be correct on any
    /// architecture.
    ///
    /// ### Why is this bad?
    /// It's basically guaranteed to be undefined behavior.
    ///
    /// ### Known problems
    /// When accessing C, users might want to store pointer
    /// sized objects in `extradata` arguments to save an allocation.
    ///
    /// ### Example
    /// ```ignore
    /// let ptr: *const T = core::intrinsics::transmute('x')
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub WRONG_TRANSMUTE,
    correctness,
    "transmutes that are confusing at best, undefined behavior at worst and always useless"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes to the original type of the object
    /// and transmutes that could be a cast.
    ///
    /// ### Why is this bad?
    /// Readability. The code tricks people into thinking that
    /// something complex is going on.
    ///
    /// ### Example
    /// ```rust,ignore
    /// core::intrinsics::transmute(t); // where the result type is the same as `t`'s
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub USELESS_TRANSMUTE,
    complexity,
    "transmutes that have the same to and from types or could be a cast/coercion"
}

// FIXME: Merge this lint with USELESS_TRANSMUTE once that is out of the nursery.
declare_clippy_lint! {
    /// ### What it does
    ///Checks for transmutes that could be a pointer cast.
    ///
    /// ### Why is this bad?
    /// Readability. The code tricks people into thinking that
    /// something complex is going on.
    ///
    /// ### Example
    ///
    /// ```no_run
    /// # let p: *const [i32] = &[];
    /// unsafe { std::mem::transmute::<*const [i32], *const [u16]>(p) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// # let p: *const [i32] = &[];
    /// p as *const [u16];
    /// ```
    #[clippy::version = "1.47.0"]
    pub TRANSMUTES_EXPRESSIBLE_AS_PTR_CASTS,
    complexity,
    "transmutes that could be a pointer cast"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes between a type `T` and `*T`.
    ///
    /// ### Why is this bad?
    /// It's easy to mistakenly transmute between a type and a
    /// pointer to that type.
    ///
    /// ### Example
    /// ```rust,ignore
    /// core::intrinsics::transmute(t) // where the result type is the same as
    ///                                // `*t` or `&t`'s
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CROSSPOINTER_TRANSMUTE,
    suspicious,
    "transmutes that have to or from types that are a pointer to the other"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes from a pointer to a reference.
    ///
    /// ### Why is this bad?
    /// This can always be rewritten with `&` and `*`.
    ///
    /// ### Known problems
    /// - `mem::transmute` in statics and constants is stable from Rust 1.46.0,
    /// while dereferencing raw pointer is not stable yet.
    /// If you need to do this in those places,
    /// you would have to use `transmute` instead.
    ///
    /// ### Example
    /// ```rust,ignore
    /// unsafe {
    ///     let _: &T = std::mem::transmute(p); // where p: *const T
    /// }
    ///
    /// // can be written:
    /// let _: &T = &*p;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TRANSMUTE_PTR_TO_REF,
    complexity,
    "transmutes from a pointer to a reference type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes from a `&[u8]` to a `&str`.
    ///
    /// ### Why is this bad?
    /// Not every byte slice is a valid UTF-8 string.
    ///
    /// ### Known problems
    /// - [`from_utf8`] which this lint suggests using is slower than `transmute`
    /// as it needs to validate the input.
    /// If you are certain that the input is always a valid UTF-8,
    /// use [`from_utf8_unchecked`] which is as fast as `transmute`
    /// but has a semantically meaningful name.
    /// - You might want to handle errors returned from [`from_utf8`] instead of calling `unwrap`.
    ///
    /// [`from_utf8`]: https://doc.rust-lang.org/std/str/fn.from_utf8.html
    /// [`from_utf8_unchecked`]: https://doc.rust-lang.org/std/str/fn.from_utf8_unchecked.html
    ///
    /// ### Example
    /// ```no_run
    /// let b: &[u8] = &[1_u8, 2_u8];
    /// unsafe {
    ///     let _: &str = std::mem::transmute(b); // where b: &[u8]
    /// }
    ///
    /// // should be:
    /// let _ = std::str::from_utf8(b).unwrap();
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TRANSMUTE_BYTES_TO_STR,
    complexity,
    "transmutes from a `&[u8]` to a `&str`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes from an integer to a `bool`.
    ///
    /// ### Why is this bad?
    /// This might result in an invalid in-memory representation of a `bool`.
    ///
    /// ### Example
    /// ```no_run
    /// let x = 1_u8;
    /// unsafe {
    ///     let _: bool = std::mem::transmute(x); // where x: u8
    /// }
    ///
    /// // should be:
    /// let _: bool = x != 0;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TRANSMUTE_INT_TO_BOOL,
    complexity,
    "transmutes from an integer to a `bool`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes from `T` to `NonZero<T>`, and suggests the `new_unchecked`
    /// method instead.
    ///
    /// ### Why is this bad?
    /// Transmutes work on any types and thus might cause unsoundness when those types change
    /// elsewhere. `new_unchecked` only works for the appropriate types instead.
    ///
    /// ### Example
    /// ```no_run
    /// # use core::num::NonZero;
    /// let _: NonZero<u32> = unsafe { std::mem::transmute(123) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// # use core::num::NonZero;
    /// let _: NonZero<u32> = unsafe { NonZero::new_unchecked(123) };
    /// ```
    #[clippy::version = "1.69.0"]
    pub TRANSMUTE_INT_TO_NON_ZERO,
    complexity,
    "transmutes from an integer to a non-zero wrapper"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes from a pointer to a pointer, or
    /// from a reference to a reference.
    ///
    /// ### Why is this bad?
    /// Transmutes are dangerous, and these can instead be
    /// written as casts.
    ///
    /// ### Example
    /// ```no_run
    /// let ptr = &1u32 as *const u32;
    /// unsafe {
    ///     // pointer-to-pointer transmute
    ///     let _: *const f32 = std::mem::transmute(ptr);
    ///     // ref-ref transmute
    ///     let _: &f32 = std::mem::transmute(&1u32);
    /// }
    /// // These can be respectively written:
    /// let _ = ptr as *const f32;
    /// let _ = unsafe{ &*(&1u32 as *const u32 as *const f32) };
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TRANSMUTE_PTR_TO_PTR,
    pedantic,
    "transmutes from a pointer to a pointer / a reference to a reference"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes between collections whose
    /// types have different ABI, size or alignment.
    ///
    /// ### Why is this bad?
    /// This is undefined behavior.
    ///
    /// ### Known problems
    /// Currently, we cannot know whether a type is a
    /// collection, so we just lint the ones that come with `std`.
    ///
    /// ### Example
    /// ```no_run
    /// // different size, therefore likely out-of-bounds memory access
    /// // You absolutely do not want this in your code!
    /// unsafe {
    ///     std::mem::transmute::<_, Vec<u32>>(vec![2_u16])
    /// };
    /// ```
    ///
    /// You must always iterate, map and collect the values:
    ///
    /// ```no_run
    /// vec![2_u16].into_iter().map(u32::from).collect::<Vec<_>>();
    /// ```
    #[clippy::version = "1.40.0"]
    pub UNSOUND_COLLECTION_TRANSMUTE,
    correctness,
    "transmute between collections of layout-incompatible types"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmutes between types which do not have a representation defined relative to
    /// each other.
    ///
    /// ### Why is this bad?
    /// The results of such a transmute are not defined.
    ///
    /// ### Known problems
    /// This lint has had multiple problems in the past and was moved to `nursery`. See issue
    /// [#8496](https://github.com/rust-lang/rust-clippy/issues/8496) for more details.
    ///
    /// ### Example
    /// ```no_run
    /// struct Foo<T>(u32, T);
    /// let _ = unsafe { core::mem::transmute::<Foo<u32>, Foo<i32>>(Foo(0u32, 0u32)) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[repr(C)]
    /// struct Foo<T>(u32, T);
    /// let _ = unsafe { core::mem::transmute::<Foo<u32>, Foo<i32>>(Foo(0u32, 0u32)) };
    /// ```
    #[clippy::version = "1.60.0"]
    pub TRANSMUTE_UNDEFINED_REPR,
    nursery,
    "transmute to or from a type with an undefined representation"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for transmute calls which would receive a null pointer.
    ///
    /// ### Why is this bad?
    /// Transmuting a null pointer is undefined behavior.
    ///
    /// ### Known problems
    /// Not all cases can be detected at the moment of this writing.
    /// For example, variables which hold a null pointer and are then fed to a `transmute`
    /// call, aren't detectable yet.
    ///
    /// ### Example
    /// ```no_run
    /// let null_ref: &u64 = unsafe { std::mem::transmute(0 as *const u64) };
    /// ```
    #[clippy::version = "1.35.0"]
    pub TRANSMUTING_NULL,
    correctness,
    "transmutes from a null pointer to a reference, which is undefined behavior"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for null function pointer creation through transmute.
    ///
    /// ### Why is this bad?
    /// Creating a null function pointer is undefined behavior.
    ///
    /// More info: https://doc.rust-lang.org/nomicon/ffi.html#the-nullable-pointer-optimization
    ///
    /// ### Known problems
    /// Not all cases can be detected at the moment of this writing.
    /// For example, variables which hold a null pointer and are then fed to a `transmute`
    /// call, aren't detectable yet.
    ///
    /// ### Example
    /// ```no_run
    /// let null_fn: fn() = unsafe { std::mem::transmute( std::ptr::null::<()>() ) };
    /// ```
    /// Use instead:
    /// ```no_run
    /// let null_fn: Option<fn()> = None;
    /// ```
    #[clippy::version = "1.68.0"]
    pub TRANSMUTE_NULL_TO_FN,
    correctness,
    "transmute results in a null function pointer, which is undefined behavior"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for integer validity checks, followed by a transmute that is (incorrectly) evaluated
    /// eagerly (e.g. using `bool::then_some`).
    ///
    /// ### Why is this bad?
    /// Eager evaluation means that the `transmute` call is executed regardless of whether the condition is true or false.
    /// This can introduce unsoundness and other subtle bugs.
    ///
    /// ### Example
    /// Consider the following function which is meant to convert an unsigned integer to its enum equivalent via transmute.
    ///
    /// ```no_run
    /// #[repr(u8)]
    /// enum Opcode {
    ///     Add = 0,
    ///     Sub = 1,
    ///     Mul = 2,
    ///     Div = 3
    /// }
    ///
    /// fn int_to_opcode(op: u8) -> Option<Opcode> {
    ///     (op < 4).then_some(unsafe { std::mem::transmute(op) })
    /// }
    /// ```
    /// This may appear fine at first given that it checks that the `u8` is within the validity range of the enum,
    /// *however* the transmute is evaluated eagerly, meaning that it executes even if `op >= 4`!
    ///
    /// This makes the function unsound, because it is possible for the caller to cause undefined behavior
    /// (creating an enum with an invalid bitpattern) entirely in safe code only by passing an incorrect value,
    /// which is normally only a bug that is possible in unsafe code.
    ///
    /// One possible way in which this can go wrong practically is that the compiler sees it as:
    /// ```rust,ignore (illustrative)
    /// let temp: Foo = unsafe { std::mem::transmute(op) };
    /// (0 < 4).then_some(temp)
    /// ```
    /// and optimizes away the `(0 < 4)` check based on the assumption that since a `Foo` was created from `op` with the validity range `0..3`,
    /// it is **impossible** for this condition to be false.
    ///
    /// In short, it is possible for this function to be optimized in a way that makes it [never return `None`](https://godbolt.org/z/ocrcenevq),
    /// even if passed the value `4`.
    ///
    /// This can be avoided by instead using lazy evaluation. For the example above, this should be written:
    /// ```rust,ignore (illustrative)
    /// fn int_to_opcode(op: u8) -> Option<Opcode> {
    ///     (op < 4).then(|| unsafe { std::mem::transmute(op) })
    ///              ^^^^ ^^ `bool::then` only executes the closure if the condition is true!
    /// }
    /// ```
    #[clippy::version = "1.77.0"]
    pub EAGER_TRANSMUTE,
    correctness,
    "eager evaluation of `transmute`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks if transmute calls have all generics specified.
    ///
    /// ### Why is this bad?
    /// If not, one or more unexpected types could be used during `transmute()`, potentially leading
    /// to Undefined Behavior or other problems.
    ///
    /// This is particularly dangerous in case a seemingly innocent/unrelated change causes type
    /// inference to result in a different type. For example, if `transmute()` is the tail
    /// expression of an `if`-branch, and the `else`-branch type changes, the compiler may silently
    /// infer a different type to be returned by `transmute()`. That is because the compiler is
    /// free to change the inference of a type as long as that inference is technically correct,
    /// regardless of the programmer's unknown expectation.
    ///
    /// Both type-parameters, the input- and the output-type, to any `transmute()` should
    /// be given explicitly: Setting the input-type explicitly avoids confusion about what the
    /// argument's type actually is. Setting the output-type explicitly avoids type-inference
    /// to infer a technically correct yet unexpected type.
    ///
    /// ### Example
    /// ```no_run
    /// # unsafe {
    /// // Avoid "naked" calls to `transmute()`!
    /// let x: i32 = std::mem::transmute([1u16, 2u16]);
    ///
    /// // `first_answers` is intended to transmute a slice of bool to a slice of u8.
    /// // But the programmer forgot to index the first element of the outer slice,
    /// // so we are actually transmuting from "pointers to slices" instead of
    /// // transmuting from "a slice of bool", causing a nonsensical result.
    /// let the_answers: &[&[bool]] = &[&[true, false, true]];
    /// let first_answers: &[u8] = std::mem::transmute(the_answers);
    /// # }
    /// ```
    /// Use instead:
    /// ```no_run
    /// # unsafe {
    /// let x = std::mem::transmute::<[u16; 2], i32>([1u16, 2u16]);
    ///
    /// // The explicit type parameters on `transmute()` makes the intention clear,
    /// // and cause a type-error if the actual types don't match our expectation.
    /// let the_answers: &[&[bool]] = &[&[true, false, true]];
    /// let first_answers: &[u8] = std::mem::transmute::<&[bool], &[u8]>(the_answers[0]);
    /// # }
    /// ```
    #[clippy::version = "1.79.0"]
    pub MISSING_TRANSMUTE_ANNOTATIONS,
    suspicious,
    "warns if a transmute call doesn't have all generics specified"
}

pub struct Transmute {
    msrv: Msrv,
}
impl_lint_pass!(Transmute => [
    CROSSPOINTER_TRANSMUTE,
    TRANSMUTE_PTR_TO_REF,
    TRANSMUTE_PTR_TO_PTR,
    USELESS_TRANSMUTE,
    WRONG_TRANSMUTE,
    TRANSMUTE_BYTES_TO_STR,
    TRANSMUTE_INT_TO_BOOL,
    TRANSMUTE_INT_TO_NON_ZERO,
    UNSOUND_COLLECTION_TRANSMUTE,
    TRANSMUTES_EXPRESSIBLE_AS_PTR_CASTS,
    TRANSMUTE_UNDEFINED_REPR,
    TRANSMUTING_NULL,
    TRANSMUTE_NULL_TO_FN,
    EAGER_TRANSMUTE,
    MISSING_TRANSMUTE_ANNOTATIONS,
]);
impl Transmute {
    pub fn new(conf: &'static Conf) -> Self {
        Self { msrv: conf.msrv }
    }
}
impl<'tcx> LateLintPass<'tcx> for Transmute {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Call(path_expr, [arg]) = e.kind
            && let ExprKind::Path(QPath::Resolved(None, path)) = path_expr.kind
            && let Some(def_id) = path.res.opt_def_id()
            && cx.tcx.is_diagnostic_item(sym::transmute, def_id)
        {
            // Avoid suggesting non-const operations in const contexts:
            // - from/to bits (https://github.com/rust-lang/rust/issues/73736)
            // - dereferencing raw pointers (https://github.com/rust-lang/rust/issues/51911)
            // - char conversions (https://github.com/rust-lang/rust/issues/89259)
            let const_context = is_in_const_context(cx);

            let (from_ty, from_ty_adjusted) = match cx.typeck_results().expr_adjustments(arg) {
                [] => (cx.typeck_results().expr_ty(arg), false),
                [.., a] => (a.target, true),
            };
            // Adjustments for `to_ty` happen after the call to `transmute`, so don't use them.
            let to_ty = cx.typeck_results().expr_ty(e);

            // If useless_transmute is triggered, the other lints can be skipped.
            if useless_transmute::check(cx, e, from_ty, to_ty, arg) {
                return;
            }

            let linted = wrong_transmute::check(cx, e, from_ty, to_ty)
                | crosspointer_transmute::check(cx, e, from_ty, to_ty)
                | transmuting_null::check(cx, e, arg, to_ty)
                | transmute_null_to_fn::check(cx, e, arg, to_ty)
                | transmute_ptr_to_ref::check(cx, e, from_ty, to_ty, arg, path, self.msrv)
                | missing_transmute_annotations::check(cx, path, arg, from_ty, to_ty, e.hir_id)
                | transmute_ref_to_ref::check(cx, e, from_ty, to_ty, arg, const_context)
                | transmute_ptr_to_ptr::check(cx, e, from_ty, to_ty, arg, self.msrv)
                | transmute_int_to_bool::check(cx, e, from_ty, to_ty, arg)
                | transmute_int_to_non_zero::check(cx, e, from_ty, to_ty, arg)
                | (unsound_collection_transmute::check(cx, e, from_ty, to_ty)
                    || transmute_undefined_repr::check(cx, e, from_ty, to_ty))
                | (eager_transmute::check(cx, e, arg, from_ty, to_ty));

            if !linted {
                transmutes_expressible_as_ptr_casts::check(cx, e, from_ty, from_ty_adjusted, to_ty, arg, const_context);
            }
        }
    }
}
