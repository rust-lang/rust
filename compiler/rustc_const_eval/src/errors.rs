use std::borrow::Cow;

use rustc_errors::codes::*;
use rustc_errors::formatting::DiagMessageAddArg;
use rustc_errors::{Diag, DiagArgValue, EmissionGuarantee, MultiSpan, Subdiagnostic, msg};
use rustc_hir::ConstContext;
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_middle::ty::{Mutability, Ty};
use rustc_span::{Span, Symbol};

use crate::interpret::InternKind;

#[derive(Diagnostic)]
#[diag(
    r#"encountered dangling pointer in final value of {$kind ->
    [static] static
    [static_mut] mutable static
    [const] constant
    [promoted] promoted
    *[other] {""}
}"#
)]
pub(crate) struct DanglingPtrInFinal {
    #[primary_span]
    pub span: Span,
    pub kind: InternKind,
}

#[derive(Diagnostic)]
#[diag(
    "#[thread_local] does not support implicit nested statics, please create explicit static items and refer to them instead"
)]
pub(crate) struct NestedStaticInThreadLocal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    r#"encountered mutable pointer in final value of {$kind ->
    [static] static
    [static_mut] mutable static
    [const] constant
    [promoted] promoted
    *[other] {""}
}"#
)]
pub(crate) struct MutablePtrInFinal {
    #[primary_span]
    pub span: Span,
    pub kind: InternKind,
}

#[derive(Diagnostic)]
#[diag("encountered `const_allocate` pointer in final value that was not made global")]
#[note(
    "use `const_make_global` to turn allocated pointers into immutable globals before returning"
)]
pub(crate) struct ConstHeapPtrInFinal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    r#"encountered partial pointer in final value of {$kind ->
    [static] static
    [static_mut] mutable static
    [const] constant
    [promoted] promoted
    *[other] {""}
}"#
)]
#[note(
    "while pointers can be broken apart into individual bytes during const-evaluation, only complete pointers (with all their bytes in the right order) are supported in the final value"
)]
pub(crate) struct PartialPtrInFinal {
    #[primary_span]
    pub span: Span,
    pub kind: InternKind,
}

#[derive(Diagnostic)]
#[diag(
    "const function that might be (indirectly) exposed to stable cannot use `#[feature({$gate})]`"
)]
pub(crate) struct UnstableInStableExposed {
    pub gate: String,
    #[primary_span]
    pub span: Span,
    #[help(
        "mark the callee as `#[rustc_const_stable_indirect]` if it does not itself require any unstable features"
    )]
    pub is_function_call: bool,
    /// Need to duplicate the field so that fluent also provides it as a variable...
    pub is_function_call2: bool,
    #[suggestion(
        "if the {$is_function_call2 ->
            [true] caller
            *[false] function
        } is not (yet) meant to be exposed to stable const contexts, add `#[rustc_const_unstable]`",
        code = "#[rustc_const_unstable(feature = \"...\", issue = \"...\")]\n",
        applicability = "has-placeholders"
    )]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("thread-local statics cannot be accessed at compile-time", code = E0625)]
pub(crate) struct ThreadLocalAccessErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("pointers cannot be cast to integers during const eval")]
#[note("at compile-time, pointers do not have an integer value")]
#[note(
    "avoiding this restriction via `transmute`, `union`, or raw pointers leads to compile-time undefined behavior"
)]
pub(crate) struct RawPtrToIntErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("pointers cannot be reliably compared during const eval")]
#[note("see issue #53020 <https://github.com/rust-lang/rust/issues/53020> for more information")]
pub(crate) struct RawPtrComparisonErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("argument to `panic!()` in a const context must have type `&str`")]
pub(crate) struct PanicNonStrErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    r#"function pointer calls are not allowed in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#
)]
pub(crate) struct UnallowedFnPointerCall {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag("`{$def_path}` is not yet stable as a const fn")]
pub(crate) struct UnstableConstFn {
    #[primary_span]
    pub span: Span,
    pub def_path: String,
}

#[derive(Diagnostic)]
#[diag("`{$def_path}` is not yet stable as a const trait")]
pub(crate) struct UnstableConstTrait {
    #[primary_span]
    pub span: Span,
    pub def_path: String,
}

#[derive(Diagnostic)]
#[diag("`{$name}` is not yet stable as a const intrinsic")]
pub(crate) struct UnstableIntrinsic {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub feature: Symbol,
    #[suggestion(
        "add `#![feature({$feature})]` to the crate attributes to enable",
        code = "#![feature({feature})]\n",
        applicability = "machine-applicable"
    )]
    pub suggestion: Span,
}

#[derive(Diagnostic)]
#[diag("`{$def_path}` cannot be (indirectly) exposed to stable")]
#[help(
    "either mark the callee as `#[rustc_const_stable_indirect]`, or the caller as `#[rustc_const_unstable]`"
)]
pub(crate) struct UnmarkedConstItemExposed {
    #[primary_span]
    pub span: Span,
    pub def_path: String,
}

#[derive(Diagnostic)]
#[diag("intrinsic `{$def_path}` cannot be (indirectly) exposed to stable")]
#[help(
    "mark the caller as `#[rustc_const_unstable]`, or mark the intrinsic `#[rustc_intrinsic_const_stable_indirect]` (but this requires team approval)"
)]
pub(crate) struct UnmarkedIntrinsicExposed {
    #[primary_span]
    pub span: Span,
    pub def_path: String,
}

#[derive(Diagnostic)]
#[diag("mutable borrows of temporaries that have their lifetime extended until the end of the program are not allowed", code = E0764)]
#[note(
    "temporaries in constants and statics can have their lifetime extended until the end of the program"
)]
#[note("to avoid accidentally creating global mutable state, such temporaries must be immutable")]
#[help(
    "if you really want global mutable state, try replacing the temporary by an interior mutable `static` or a `static mut`"
)]
pub(crate) struct MutableBorrowEscaping {
    #[primary_span]
    #[label("this mutable borrow refers to such a temporary")]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(
    r#"cannot call {$non_or_conditionally}-const formatting macro in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#,
    code = E0015,
)]
pub(crate) struct NonConstFmtMacroCall {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"cannot call {$non_or_conditionally}-const {$def_descr} `{$def_path_str}` in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstFnCall {
    #[primary_span]
    pub span: Span,
    pub def_path_str: String,
    pub def_descr: &'static str,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(
    r#"cannot call non-const intrinsic `{$name}` in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#
)]
pub(crate) struct NonConstIntrinsic {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag("{$msg}")]
pub(crate) struct UnallowedOpInConstContext {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}

#[derive(Diagnostic)]
#[diag(r#"inline assembly is not allowed in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct UnallowedInlineAsm {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag("interior mutable shared borrows of temporaries that have their lifetime extended until the end of the program are not allowed", code = E0492)]
#[note(
    "temporaries in constants and statics can have their lifetime extended until the end of the program"
)]
#[note("to avoid accidentally creating global mutable state, such temporaries must be immutable")]
#[help(
    "if you really want global mutable state, try replacing the temporary by an interior mutable `static` or a `static mut`"
)]
pub(crate) struct InteriorMutableBorrowEscaping {
    #[primary_span]
    #[label("this borrow of an interior mutable value refers to such a temporary")]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag("constant evaluation is taking a long time")]
#[note(
    "this lint makes sure the compiler doesn't get stuck due to infinite loops in const eval.
    If your compilation actually takes a long time, you can safely allow the lint"
)]
pub(crate) struct LongRunning {
    #[help("the constant being evaluated")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("constant evaluation is taking a long time")]
pub(crate) struct LongRunningWarn {
    #[primary_span]
    #[label("the const evaluator is currently interpreting this expression")]
    pub span: Span,
    #[help("the constant being evaluated")]
    pub item_span: Span,
    // Used for evading `-Z deduplicate-diagnostics`.
    pub force_duplicate: usize,
}

#[derive(Subdiagnostic)]
#[note("impl defined here, but it is not `const`")]
pub(crate) struct NonConstImplNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Clone)]
pub(crate) struct FrameNote {
    pub span: Span,
    pub times: i32,
    pub where_: &'static str,
    pub instance: String,
    pub has_label: bool,
}

impl Subdiagnostic for FrameNote {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        let mut span: MultiSpan = self.span.into();
        if self.has_label && !self.span.is_dummy() {
            span.push_span_label(self.span, msg!("the failure occurred here"));
        }
        let msg = msg!(
            r#"{$times ->
                [0] inside {$where_ ->
                    [closure] closure
                    [instance] `{$instance}`
                    *[other] {""}
                }
                *[other] [... {$times} additional calls inside {$where_ ->
                    [closure] closure
                    [instance] `{$instance}`
                    *[other] {""}
                } ...]
            }"#
        )
        .arg("times", self.times)
        .arg("where_", self.where_)
        .arg("instance", self.instance)
        .format();
        diag.span_note(span, msg);
    }
}

#[derive(Subdiagnostic)]
#[note(r#"the raw bytes of the constant (size: {$size}, align: {$align}) {"{"}{$bytes}{"}"}"#)]
pub(crate) struct RawBytesNote {
    pub size: u64,
    pub align: u64,
    pub bytes: String,
}

#[derive(Diagnostic)]
#[diag(
    r#"cannot match on `{$ty}` in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#
)]
#[note("`{$ty}` cannot be compared in compile-time, and therefore cannot be used in `match`es")]
pub(crate) struct NonConstMatchEq<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"cannot use `for` loop on `{$ty}` in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstForLoopIntoIter<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"`?` is not allowed on `{$ty}` in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstQuestionBranch<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"`?` is not allowed on `{$ty}` in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstQuestionFromResidual<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"`try` block cannot convert `{$ty}` to the result in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstTryBlockFromOutput<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"cannot convert `{$ty}` into a future in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstAwait<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"cannot call {$non_or_conditionally}-const closure in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstClosure {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[subdiagnostic]
    pub note: Option<NonConstClosureNote>,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"calling const c-variadic functions is unstable in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstCVariadicCall {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Subdiagnostic)]
pub(crate) enum NonConstClosureNote {
    #[note("function defined here, but it is not `const`")]
    FnDef {
        #[primary_span]
        span: Span,
    },
    #[note(
        r#"function pointers need an RFC before allowed to be called in {$kind ->
            [const] constant
            [static] static
            [const_fn] constant function
            *[other] {""}
        }s"#
    )]
    FnPtr { kind: ConstContext },
    #[note(
        r#"closures need an RFC before allowed to be called in {$kind ->
            [const] constant
            [static] static
            [const_fn] constant function
            *[other] {""}
        }s"#
    )]
    Closure { kind: ConstContext },
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("consider dereferencing here", applicability = "machine-applicable")]
pub(crate) struct ConsiderDereferencing {
    pub deref: String,
    #[suggestion_part(code = "{deref}")]
    pub span: Span,
    #[suggestion_part(code = "{deref}")]
    pub rhs_span: Span,
}

#[derive(Diagnostic)]
#[diag(r#"cannot call {$non_or_conditionally}-const operator in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
pub(crate) struct NonConstOperator {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[subdiagnostic]
    pub sugg: Option<ConsiderDereferencing>,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag(r#"cannot perform {$non_or_conditionally}-const deref coercion on `{$ty}` in {$kind ->
    [const] constant
    [static] static
    [const_fn] constant function
    *[other] {""}
}s"#, code = E0015)]
#[note("attempting to deref into `{$target_ty}`")]
pub(crate) struct NonConstDerefCoercion<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub target_ty: Ty<'tcx>,
    #[note("deref defined here")]
    pub deref_target: Option<Span>,
    pub non_or_conditionally: &'static str,
}

#[derive(Diagnostic)]
#[diag("destructor of `{$dropped_ty}` cannot be evaluated at compile-time", code = E0493)]
pub(crate) struct LiveDrop<'tcx> {
    #[primary_span]
    #[label(
        r#"the destructor for this type cannot be evaluated in {$kind ->
            [const] constant
            [static] static
            [const_fn] constant function
            *[other] {""}
        }s"#
    )]
    pub span: Span,
    pub kind: ConstContext,
    pub dropped_ty: Ty<'tcx>,
    #[label("value is dropped here")]
    pub dropped_at: Span,
}

impl rustc_errors::IntoDiagArg for InternKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(match self {
            InternKind::Static(Mutability::Not) => "static",
            InternKind::Static(Mutability::Mut) => "static_mut",
            InternKind::Constant => "const",
            InternKind::Promoted => "promoted",
        }))
    }
}
