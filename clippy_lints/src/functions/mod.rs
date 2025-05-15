mod impl_trait_in_params;
mod misnamed_getters;
mod must_use;
mod not_unsafe_ptr_arg_deref;
mod ref_option;
mod renamed_function_params;
mod result;
mod too_many_arguments;
mod too_many_lines;

use clippy_config::Conf;
use clippy_utils::msrvs::Msrv;
use clippy_utils::paths::{PathNS, lookup_path_str};
use rustc_hir as hir;
use rustc_hir::intravisit;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::TyCtxt;
use rustc_session::impl_lint_pass;
use rustc_span::Span;
use rustc_span::def_id::{DefIdSet, LocalDefId};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions with too many parameters.
    ///
    /// ### Why is this bad?
    /// Functions with lots of parameters are considered bad
    /// style and reduce readability (“what does the 5th parameter mean?”). Consider
    /// grouping some parameters into a new type.
    ///
    /// ### Example
    /// ```no_run
    /// # struct Color;
    /// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b: f32) {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub TOO_MANY_ARGUMENTS,
    complexity,
    "functions with too many arguments"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions with a large amount of lines.
    ///
    /// ### Why is this bad?
    /// Functions with a lot of lines are harder to understand
    /// due to having to look at a larger amount of code to understand what the
    /// function is doing. Consider splitting the body of the function into
    /// multiple functions.
    ///
    /// ### Example
    /// ```no_run
    /// fn im_too_long() {
    ///     println!("");
    ///     // ... 100 more LoC
    ///     println!("");
    /// }
    /// ```
    #[clippy::version = "1.34.0"]
    pub TOO_MANY_LINES,
    pedantic,
    "functions with too many lines"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public functions that dereference raw pointer
    /// arguments but are not marked `unsafe`.
    ///
    /// ### Why is this bad?
    /// The function should almost definitely be marked `unsafe`, since for an
    /// arbitrary raw pointer, there is no way of telling for sure if it is valid.
    ///
    /// In general, this lint should **never be disabled** unless it is definitely a
    /// false positive (please submit an issue if so) since it breaks Rust's
    /// soundness guarantees, directly exposing API users to potentially dangerous
    /// program behavior. This is also true for internal APIs, as it is easy to leak
    /// unsoundness.
    ///
    /// ### Context
    /// In Rust, an `unsafe {...}` block is used to indicate that the code in that
    /// section has been verified in some way that the compiler can not. For a
    /// function that accepts a raw pointer then accesses the pointer's data, this is
    /// generally impossible as the incoming pointer could point anywhere, valid or
    /// not. So, the signature should be marked `unsafe fn`: this indicates that the
    /// function's caller must provide some verification that the arguments it sends
    /// are valid (and then call the function within an `unsafe` block).
    ///
    /// ### Known problems
    /// * It does not check functions recursively so if the pointer is passed to a
    /// private non-`unsafe` function which does the dereferencing, the lint won't
    /// trigger (false negative).
    /// * It only checks for arguments whose type are raw pointers, not raw pointers
    /// got from an argument in some other way (`fn foo(bar: &[*const u8])` or
    /// `some_argument.get_raw_ptr()`) (false negative).
    ///
    /// ### Example
    /// ```rust,ignore
    /// pub fn foo(x: *const u8) {
    ///     println!("{}", unsafe { *x });
    /// }
    ///
    /// // this call "looks" safe but will segfault or worse!
    /// // foo(invalid_ptr);
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// pub unsafe fn foo(x: *const u8) {
    ///     println!("{}", unsafe { *x });
    /// }
    ///
    /// // this would cause a compiler error for calling without `unsafe`
    /// // foo(invalid_ptr);
    ///
    /// // sound call if the caller knows the pointer is valid
    /// unsafe { foo(valid_ptr); }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NOT_UNSAFE_PTR_ARG_DEREF,
    correctness,
    "public functions dereferencing raw pointer arguments but not marked `unsafe`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a `#[must_use]` attribute on
    /// unit-returning functions and methods.
    ///
    /// ### Why is this bad?
    /// Unit values are useless. The attribute is likely
    /// a remnant of a refactoring that removed the return type.
    ///
    /// ### Examples
    /// ```no_run
    /// #[must_use]
    /// fn useless() { }
    /// ```
    #[clippy::version = "1.40.0"]
    pub MUST_USE_UNIT,
    style,
    "`#[must_use]` attribute on a unit-returning function / method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for a `#[must_use]` attribute without
    /// further information on functions and methods that return a type already
    /// marked as `#[must_use]`.
    ///
    /// ### Why is this bad?
    /// The attribute isn't needed. Not using the result
    /// will already be reported. Alternatively, one can add some text to the
    /// attribute to improve the lint message.
    ///
    /// ### Examples
    /// ```no_run
    /// #[must_use]
    /// fn double_must_use() -> Result<(), ()> {
    ///     unimplemented!();
    /// }
    /// ```
    #[clippy::version = "1.40.0"]
    pub DOUBLE_MUST_USE,
    style,
    "`#[must_use]` attribute on a `#[must_use]`-returning function / method"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public functions that have no
    /// `#[must_use]` attribute, but return something not already marked
    /// must-use, have no mutable arg and mutate no statics.
    ///
    /// ### Why is this bad?
    /// Not bad at all, this lint just shows places where
    /// you could add the attribute.
    ///
    /// ### Known problems
    /// The lint only checks the arguments for mutable
    /// types without looking if they are actually changed. On the other hand,
    /// it also ignores a broad range of potentially interesting side effects,
    /// because we cannot decide whether the programmer intends the function to
    /// be called for the side effect or the result. Expect many false
    /// positives. At least we don't lint if the result type is unit or already
    /// `#[must_use]`.
    ///
    /// ### Examples
    /// ```no_run
    /// // this could be annotated with `#[must_use]`.
    /// pub fn id<T>(t: T) -> T { t }
    /// ```
    #[clippy::version = "1.40.0"]
    pub MUST_USE_CANDIDATE,
    pedantic,
    "function or method that could take a `#[must_use]` attribute"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for public functions that return a `Result`
    /// with an `Err` type of `()`. It suggests using a custom type that
    /// implements `std::error::Error`.
    ///
    /// ### Why is this bad?
    /// Unit does not implement `Error` and carries no
    /// further information about what went wrong.
    ///
    /// ### Known problems
    /// Of course, this lint assumes that `Result` is used
    /// for a fallible operation (which is after all the intended use). However
    /// code may opt to (mis)use it as a basic two-variant-enum. In that case,
    /// the suggestion is misguided, and the code should use a custom enum
    /// instead.
    ///
    /// ### Examples
    /// ```no_run
    /// pub fn read_u8() -> Result<u8, ()> { Err(()) }
    /// ```
    /// should become
    /// ```rust,should_panic
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// pub struct EndOfStream;
    ///
    /// impl fmt::Display for EndOfStream {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "End of Stream")
    ///     }
    /// }
    ///
    /// impl std::error::Error for EndOfStream { }
    ///
    /// pub fn read_u8() -> Result<u8, EndOfStream> { Err(EndOfStream) }
    ///# fn main() {
    ///#     read_u8().unwrap();
    ///# }
    /// ```
    ///
    /// Note that there are crates that simplify creating the error type, e.g.
    /// [`thiserror`](https://docs.rs/thiserror).
    #[clippy::version = "1.49.0"]
    pub RESULT_UNIT_ERR,
    style,
    "public function returning `Result` with an `Err` type of `()`"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for functions that return `Result` with an unusually large
    /// `Err`-variant.
    ///
    /// ### Why is this bad?
    /// A `Result` is at least as large as the `Err`-variant. While we
    /// expect that variant to be seldom used, the compiler needs to reserve
    /// and move that much memory every single time.
    /// Furthermore, errors are often simply passed up the call-stack, making
    /// use of the `?`-operator and its type-conversion mechanics. If the
    /// `Err`-variant further up the call-stack stores the `Err`-variant in
    /// question (as library code often does), it itself needs to be at least
    /// as large, propagating the problem.
    ///
    /// ### Known problems
    /// The size determined by Clippy is platform-dependent.
    ///
    /// ### Examples
    /// ```no_run
    /// pub enum ParseError {
    ///     UnparsedBytes([u8; 512]),
    ///     UnexpectedEof,
    /// }
    ///
    /// // The `Result` has at least 512 bytes, even in the `Ok`-case
    /// pub fn parse() -> Result<(), ParseError> {
    ///     Ok(())
    /// }
    /// ```
    /// should be
    /// ```no_run
    /// pub enum ParseError {
    ///     UnparsedBytes(Box<[u8; 512]>),
    ///     UnexpectedEof,
    /// }
    ///
    /// // The `Result` is slightly larger than a pointer
    /// pub fn parse() -> Result<(), ParseError> {
    ///     Ok(())
    /// }
    /// ```
    #[clippy::version = "1.65.0"]
    pub RESULT_LARGE_ERR,
    perf,
    "function returning `Result` with large `Err` type"
}

declare_clippy_lint! {
    /// ### What it does
    /// Checks for getter methods that return a field that doesn't correspond
    /// to the name of the method, when there is a field's whose name matches that of the method.
    ///
    /// ### Why is this bad?
    /// It is most likely that such a  method is a bug caused by a typo or by copy-pasting.
    ///
    /// ### Example

    /// ```no_run
    /// struct A {
    ///     a: String,
    ///     b: String,
    /// }
    ///
    /// impl A {
    ///     fn a(&self) -> &str{
    ///         &self.b
    ///     }
    /// }

    /// ```
    /// Use instead:
    /// ```no_run
    /// struct A {
    ///     a: String,
    ///     b: String,
    /// }
    ///
    /// impl A {
    ///     fn a(&self) -> &str{
    ///         &self.a
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.67.0"]
    pub MISNAMED_GETTERS,
    suspicious,
    "getter method returning the wrong field"
}

declare_clippy_lint! {
    /// ### What it does
    /// Lints when `impl Trait` is being used in a function's parameters.
    ///
    /// ### Why restrict this?
    /// Turbofish syntax (`::<>`) cannot be used to specify the type of an `impl Trait` parameter,
    /// making `impl Trait` less powerful. Readability may also be a factor.
    ///
    /// ### Example
    /// ```no_run
    /// trait MyTrait {}
    /// fn foo(a: impl MyTrait) {
    /// 	// [...]
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// trait MyTrait {}
    /// fn foo<T: MyTrait>(a: T) {
    /// 	// [...]
    /// }
    /// ```
    #[clippy::version = "1.69.0"]
    pub IMPL_TRAIT_IN_PARAMS,
    restriction,
    "`impl Trait` is used in the function's parameters"
}

declare_clippy_lint! {
    /// ### What it does
    /// Lints when the name of function parameters from trait impl is
    /// different than its default implementation.
    ///
    /// ### Why restrict this?
    /// Using the default name for parameters of a trait method is more consistent.
    ///
    /// ### Example
    /// ```rust
    /// struct A(u32);
    ///
    /// impl PartialEq for A {
    ///     fn eq(&self, b: &Self) -> bool {
    ///         self.0 == b.0
    ///     }
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// struct A(u32);
    ///
    /// impl PartialEq for A {
    ///     fn eq(&self, other: &Self) -> bool {
    ///         self.0 == other.0
    ///     }
    /// }
    /// ```
    #[clippy::version = "1.80.0"]
    pub RENAMED_FUNCTION_PARAMS,
    restriction,
    "renamed function parameters in trait implementation"
}

declare_clippy_lint! {
    /// ### What it does
    /// Warns when a function signature uses `&Option<T>` instead of `Option<&T>`.
    ///
    /// ### Why is this bad?
    /// More flexibility, better memory optimization, and more idiomatic Rust code.
    ///
    /// `&Option<T>` in a function signature breaks encapsulation because the caller must own T
    /// and move it into an Option to call with it. When returned, the owner must internally store
    /// it as `Option<T>` in order to return it.
    /// At a lower level, `&Option<T>` points to memory with the `presence` bit flag plus the `T` value,
    /// whereas `Option<&T>` is usually [optimized](https://doc.rust-lang.org/1.81.0/std/option/index.html#representation)
    /// to a single pointer, so it may be more optimal.
    ///
    /// See this [YouTube video](https://www.youtube.com/watch?v=6c7pZYP_iIE) by
    /// Logan Smith for an in-depth explanation of why this is important.
    ///
    /// ### Known problems
    /// This lint recommends changing the function signatures, but it cannot
    /// automatically change the function calls or the function implementations.
    ///
    /// ### Example
    /// ```no_run
    /// // caller uses  foo(&opt)
    /// fn foo(a: &Option<String>) {}
    /// # struct Unit {}
    /// # impl Unit {
    /// fn bar(&self) -> &Option<String> { &None }
    /// # }
    /// ```
    /// Use instead:
    /// ```no_run
    /// // caller should use  `foo1(opt.as_ref())`
    /// fn foo1(a: Option<&String>) {}
    /// // better yet, use string slice  `foo2(opt.as_deref())`
    /// fn foo2(a: Option<&str>) {}
    /// # struct Unit {}
    /// # impl Unit {
    /// fn bar(&self) -> Option<&String> { None }
    /// # }
    /// ```
    #[clippy::version = "1.83.0"]
    pub REF_OPTION,
    pedantic,
    "function signature uses `&Option<T>` instead of `Option<&T>`"
}

pub struct Functions {
    too_many_arguments_threshold: u64,
    too_many_lines_threshold: u64,
    large_error_threshold: u64,
    avoid_breaking_exported_api: bool,
    /// A set of resolved `def_id` of traits that are configured to allow
    /// function params renaming.
    trait_ids: DefIdSet,
    msrv: Msrv,
}

impl Functions {
    pub fn new(tcx: TyCtxt<'_>, conf: &'static Conf) -> Self {
        Self {
            too_many_arguments_threshold: conf.too_many_arguments_threshold,
            too_many_lines_threshold: conf.too_many_lines_threshold,
            large_error_threshold: conf.large_error_threshold,
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
            trait_ids: conf
                .allow_renamed_params_for
                .iter()
                .flat_map(|p| lookup_path_str(tcx, PathNS::Type, p))
                .collect(),
            msrv: conf.msrv,
        }
    }
}

impl_lint_pass!(Functions => [
    TOO_MANY_ARGUMENTS,
    TOO_MANY_LINES,
    NOT_UNSAFE_PTR_ARG_DEREF,
    MUST_USE_UNIT,
    DOUBLE_MUST_USE,
    MUST_USE_CANDIDATE,
    RESULT_UNIT_ERR,
    RESULT_LARGE_ERR,
    MISNAMED_GETTERS,
    IMPL_TRAIT_IN_PARAMS,
    RENAMED_FUNCTION_PARAMS,
    REF_OPTION,
]);

impl<'tcx> LateLintPass<'tcx> for Functions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        body: &'tcx hir::Body<'_>,
        span: Span,
        def_id: LocalDefId,
    ) {
        let hir_id = cx.tcx.local_def_id_to_hir_id(def_id);
        too_many_arguments::check_fn(cx, kind, decl, span, hir_id, self.too_many_arguments_threshold);
        too_many_lines::check_fn(cx, kind, span, body, self.too_many_lines_threshold);
        not_unsafe_ptr_arg_deref::check_fn(cx, kind, decl, body, def_id);
        misnamed_getters::check_fn(cx, kind, decl, body, span);
        impl_trait_in_params::check_fn(cx, &kind, body, hir_id);
        ref_option::check_fn(
            cx,
            kind,
            decl,
            span,
            hir_id,
            def_id,
            body,
            self.avoid_breaking_exported_api,
        );
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        must_use::check_item(cx, item);
        result::check_item(cx, item, self.large_error_threshold, self.msrv);
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        must_use::check_impl_item(cx, item);
        result::check_impl_item(cx, item, self.large_error_threshold, self.msrv);
        impl_trait_in_params::check_impl_item(cx, item);
        renamed_function_params::check_impl_item(cx, item, &self.trait_ids);
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        too_many_arguments::check_trait_item(cx, item, self.too_many_arguments_threshold);
        not_unsafe_ptr_arg_deref::check_trait_item(cx, item);
        must_use::check_trait_item(cx, item);
        result::check_trait_item(cx, item, self.large_error_threshold, self.msrv);
        impl_trait_in_params::check_trait_item(cx, item, self.avoid_breaking_exported_api);
        ref_option::check_trait_item(cx, item, self.avoid_breaking_exported_api);
    }
}
