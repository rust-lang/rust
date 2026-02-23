use std::borrow::Cow;
use std::fmt::Write;

use either::Either;
use rustc_abi::WrappingRange;
use rustc_errors::codes::*;
use rustc_errors::{
    Diag, DiagArgValue, DiagMessage, Diagnostic, EmissionGuarantee, Level, MultiSpan,
    Subdiagnostic, msg,
};
use rustc_hir::ConstContext;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::interpret::{
    CtfeProvenance, ExpectedKind, InterpErrorKind, InvalidMetaKind, InvalidProgramInfo,
    Misalignment, Pointer, PointerKind, ResourceExhaustionInfo, UndefinedBehaviorInfo,
    UnsupportedOpInfo, ValidationErrorInfo,
};
use rustc_middle::ty::{self, Mutability, Ty};
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

#[derive(LintDiagnostic)]
#[diag("constant evaluation is taking a long time")]
#[note(
    "this lint makes sure the compiler doesn't get stuck due to infinite loops in const eval.
    If your compilation actually takes a long time, you can safely allow the lint"
)]
pub struct LongRunning {
    #[help("the constant being evaluated")]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag("constant evaluation is taking a long time")]
pub struct LongRunningWarn {
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
pub struct FrameNote {
    pub span: Span,
    pub times: i32,
    pub where_: &'static str,
    pub instance: String,
    pub has_label: bool,
}

impl Subdiagnostic for FrameNote {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.arg("times", self.times);
        diag.arg("where_", self.where_);
        diag.arg("instance", self.instance);
        let mut span: MultiSpan = self.span.into();
        if self.has_label && !self.span.is_dummy() {
            span.push_span_label(self.span, msg!("the failure occurred here"));
        }
        let msg = diag.eagerly_translate(msg!(
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
        ));
        diag.remove_arg("times");
        diag.remove_arg("where_");
        diag.remove_arg("instance");
        diag.span_note(span, msg);
    }
}

#[derive(Subdiagnostic)]
#[note(r#"the raw bytes of the constant (size: {$size}, align: {$align}) {"{"}{$bytes}{"}"}"#)]
pub struct RawBytesNote {
    pub size: u64,
    pub align: u64,
    pub bytes: String,
}

// FIXME(fee1-dead) do not use stringly typed `ConstContext`

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
pub struct NonConstMatchEq<'tcx> {
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
pub struct NonConstForLoopIntoIter<'tcx> {
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
pub struct NonConstQuestionBranch<'tcx> {
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
pub struct NonConstQuestionFromResidual<'tcx> {
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
pub struct NonConstTryBlockFromOutput<'tcx> {
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
pub struct NonConstAwait<'tcx> {
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
pub struct NonConstClosure {
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
pub struct NonConstCVariadicCall {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Subdiagnostic)]
pub enum NonConstClosureNote {
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
    FnPtr,
    #[note(
        r#"closures need an RFC before allowed to be called in {$kind ->
            [const] constant
            [static] static
            [const_fn] constant function
            *[other] {""}
        }s"#
    )]
    Closure,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion("consider dereferencing here", applicability = "machine-applicable")]
pub struct ConsiderDereferencing {
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
pub struct NonConstOperator {
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
pub struct NonConstDerefCoercion<'tcx> {
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
pub struct LiveDrop<'tcx> {
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

pub trait ReportErrorExt {
    /// Returns the diagnostic message for this error.
    fn diagnostic_message(&self) -> DiagMessage;
    fn add_args<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>);

    fn debug(self) -> String
    where
        Self: Sized,
    {
        ty::tls::with(move |tcx| {
            let dcx = tcx.dcx();
            let mut diag = dcx.struct_allow(DiagMessage::Str(String::new().into()));
            let message = self.diagnostic_message();
            self.add_args(&mut diag);
            let s = dcx.eagerly_translate_to_string(message, diag.args.iter());
            diag.cancel();
            s
        })
    }
}

impl<'a> ReportErrorExt for UndefinedBehaviorInfo<'a> {
    fn diagnostic_message(&self) -> DiagMessage {
        use UndefinedBehaviorInfo::*;

        match self {
            Ub(msg) => msg.clone().into(),
            Custom(x) => (x.msg)(),
            ValidationError(e) => e.diagnostic_message(),

            Unreachable => "entering unreachable code".into(),
            BoundsCheckFailed { .. } => msg!("indexing out of bounds: the len is {$len} but the index is {$index}"),
            DivisionByZero => "dividing by zero".into(),
            RemainderByZero => "calculating the remainder with a divisor of zero".into(),
            DivisionOverflow => "overflow in signed division (dividing MIN by -1)".into(),
            RemainderOverflow => "overflow in signed remainder (dividing MIN by -1)".into(),
            PointerArithOverflow => "overflowing pointer arithmetic: the total offset in bytes does not fit in an `isize`".into(),
            ArithOverflow { .. } => msg!("arithmetic overflow in `{$intrinsic}`"),
            ShiftOverflow { .. } => msg!("overflowing shift by {$shift_amount} in `{$intrinsic}`"),
            InvalidMeta(InvalidMetaKind::SliceTooBig) => "invalid metadata in wide pointer: slice is bigger than largest supported object".into(),
            InvalidMeta(InvalidMetaKind::TooBig) => "invalid metadata in wide pointer: total size is bigger than largest supported object".into(),
            UnterminatedCString(_) => "reading a null-terminated string starting at {$pointer} with no null found before end of allocation".into(),
            PointerUseAfterFree(_, _) => msg!(
                "{$operation ->
                    [MemoryAccess] memory access failed
                    [InboundsPointerArithmetic] in-bounds pointer arithmetic failed
                    *[Dereferenceable] pointer not dereferenceable
                }: {$alloc_id} has been freed, so this pointer is dangling"
            ),
            PointerOutOfBounds { .. } => msg!(
                "{$operation ->
                    [MemoryAccess] memory access failed
                    [InboundsPointerArithmetic] in-bounds pointer arithmetic failed
                    *[Dereferenceable] pointer not dereferenceable
                }: {$operation ->
                    [MemoryAccess] attempting to access {$inbounds_size ->
                        [1] 1 byte
                        *[x] {$inbounds_size} bytes
                    }
                    [InboundsPointerArithmetic] attempting to offset pointer by {$inbounds_size ->
                        [1] 1 byte
                        *[x] {$inbounds_size} bytes
                    }
                    *[Dereferenceable] pointer must {$inbounds_size ->
                        [0] point to some allocation
                        [1] be dereferenceable for 1 byte
                        *[x] be dereferenceable for {$inbounds_size} bytes
                    }
                }, but got {$pointer} which {$ptr_offset_is_neg ->
                    [true] points to before the beginning of the allocation
                    *[false] {$inbounds_size_is_neg ->
                        [false] {$alloc_size_minus_ptr_offset ->
                            [0] is at or beyond the end of the allocation of size {$alloc_size ->
                                [1] 1 byte
                                *[x] {$alloc_size} bytes
                            }
                            [1] is only 1 byte from the end of the allocation
                            *[x] is only {$alloc_size_minus_ptr_offset} bytes from the end of the allocation
                        }
                        *[true] {$ptr_offset_abs ->
                            [0] is at the beginning of the allocation
                            *[other] is only {$ptr_offset_abs} bytes from the beginning of the allocation
                        }
                    }
                }"
            ),
            DanglingIntPointer { addr: 0, .. } => msg!(
                "{$operation ->
                    [MemoryAccess] memory access failed
                    [InboundsPointerArithmetic] in-bounds pointer arithmetic failed
                    *[Dereferenceable] pointer not dereferenceable
                }: {$operation ->
                    [MemoryAccess] attempting to access {$inbounds_size ->
                        [1] 1 byte
                        *[x] {$inbounds_size} bytes
                    }
                    [InboundsPointerArithmetic] attempting to offset pointer by {$inbounds_size ->
                        [1] 1 byte
                        *[x] {$inbounds_size} bytes
                    }
                    *[Dereferenceable] pointer must {$inbounds_size ->
                        [0] point to some allocation
                        [1] be dereferenceable for 1 byte
                        *[x] be dereferenceable for {$inbounds_size} bytes
                    }
                }, but got null pointer"),
            DanglingIntPointer { .. } => msg!(
                "{$operation ->
                    [MemoryAccess] memory access failed
                    [InboundsPointerArithmetic] in-bounds pointer arithmetic failed
                    *[Dereferenceable] pointer not dereferenceable
                }: {$operation ->
                    [MemoryAccess] attempting to access {$inbounds_size ->
                        [1] 1 byte
                        *[x] {$inbounds_size} bytes
                    }
                    [InboundsPointerArithmetic] attempting to offset pointer by {$inbounds_size ->
                        [1] 1 byte
                        *[x] {$inbounds_size} bytes
                    }
                    *[Dereferenceable] pointer must {$inbounds_size ->
                        [0] point to some allocation
                        [1] be dereferenceable for 1 byte
                        *[x] be dereferenceable for {$inbounds_size} bytes
                    }
                }, but got {$pointer} which is a dangling pointer (it has no provenance)"),
            AlignmentCheckFailed { .. } => msg!(
                "{$msg ->
                    [AccessedPtr] accessing memory
                    *[other] accessing memory based on pointer
                } with alignment {$has}, but alignment {$required} is required"
            ),
            WriteToReadOnly(_) => msg!("writing to {$allocation} which is read-only"),
            DerefFunctionPointer(_) => msg!("accessing {$allocation} which contains a function"),
            DerefVTablePointer(_) => msg!("accessing {$allocation} which contains a vtable"),
            DerefVaListPointer(_) => msg!("accessing {$allocation} which contains a variable argument list"),
            DerefTypeIdPointer(_) => msg!("accessing {$allocation} which contains a `TypeId`"),
            InvalidBool(_) => msg!("interpreting an invalid 8-bit value as a bool: 0x{$value}"),
            InvalidChar(_) => msg!("interpreting an invalid 32-bit value as a char: 0x{$value}"),
            InvalidTag(_) => msg!("enum value has invalid tag: {$tag}"),
            InvalidFunctionPointer(_) => msg!("using {$pointer} as function pointer but it does not point to a function"),
            InvalidVaListPointer(_) => msg!("using {$pointer} as variable argument list pointer but it does not point to a variable argument list"),
            InvalidVTablePointer(_) => msg!("using {$pointer} as vtable pointer but it does not point to a vtable"),
            InvalidVTableTrait { .. } => msg!("using vtable for `{$vtable_dyn_type}` but `{$expected_dyn_type}` was expected"),
            InvalidStr(_) => msg!("this string is not valid UTF-8: {$err}"),
            InvalidUninitBytes(None) => "using uninitialized data, but this operation requires initialized memory".into(),
            InvalidUninitBytes(Some(_)) => msg!("reading memory at {$alloc}{$access}, but memory is uninitialized at {$uninit}, and this operation requires initialized memory"),
            DeadLocal => "accessing a dead local variable".into(),
            ScalarSizeMismatch(_) => msg!("scalar size mismatch: expected {$target_size} bytes but got {$data_size} bytes instead"),
            UninhabitedEnumVariantWritten(_) => "writing discriminant of an uninhabited enum variant".into(),
            UninhabitedEnumVariantRead(_) => "read discriminant of an uninhabited enum variant".into(),
            InvalidNichedEnumVariantWritten { .. } => {
                msg!("trying to set discriminant of a {$ty} to the niched variant, but the value does not match")
            }
            AbiMismatchArgument { .. } => msg!("calling a function whose parameter #{$arg_idx} has type {$callee_ty} passing argument of type {$caller_ty}"),
            AbiMismatchReturn { .. } => msg!("calling a function with return type {$callee_ty} passing return place of type {$caller_ty}"),
            VaArgOutOfBounds => "more C-variadic arguments read than were passed".into(),
            CVariadicMismatch { ..} => "calling a function where the caller and callee disagree on whether the function is C-variadic".into(),
            CVariadicFixedCountMismatch { .. } => msg!("calling a C-variadic function with {$caller} fixed arguments, but the function expects {$callee}"),
        }
    }

    fn add_args<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        use UndefinedBehaviorInfo::*;
        match self {
            Ub(_) => {}
            Custom(custom) => {
                (custom.add_args)(&mut |name, value| {
                    diag.arg(name, value);
                });
            }
            ValidationError(e) => e.add_args(diag),

            Unreachable
            | DivisionByZero
            | RemainderByZero
            | DivisionOverflow
            | RemainderOverflow
            | PointerArithOverflow
            | InvalidMeta(InvalidMetaKind::SliceTooBig)
            | InvalidMeta(InvalidMetaKind::TooBig)
            | InvalidUninitBytes(None)
            | DeadLocal
            | VaArgOutOfBounds
            | UninhabitedEnumVariantWritten(_)
            | UninhabitedEnumVariantRead(_) => {}

            ArithOverflow { intrinsic } => {
                diag.arg("intrinsic", intrinsic);
            }
            ShiftOverflow { intrinsic, shift_amount } => {
                diag.arg("intrinsic", intrinsic);
                diag.arg(
                    "shift_amount",
                    match shift_amount {
                        Either::Left(v) => v.to_string(),
                        Either::Right(v) => v.to_string(),
                    },
                );
            }
            BoundsCheckFailed { len, index } => {
                diag.arg("len", len);
                diag.arg("index", index);
            }
            UnterminatedCString(ptr)
            | InvalidFunctionPointer(ptr)
            | InvalidVaListPointer(ptr)
            | InvalidVTablePointer(ptr) => {
                diag.arg("pointer", ptr);
            }
            InvalidVTableTrait { expected_dyn_type, vtable_dyn_type } => {
                diag.arg("expected_dyn_type", expected_dyn_type.to_string());
                diag.arg("vtable_dyn_type", vtable_dyn_type.to_string());
            }
            PointerUseAfterFree(alloc_id, msg) => {
                diag.arg("alloc_id", alloc_id).arg("operation", format!("{:?}", msg));
            }
            PointerOutOfBounds { alloc_id, alloc_size, ptr_offset, inbounds_size, msg } => {
                diag.arg("alloc_size", alloc_size.bytes());
                diag.arg("pointer", {
                    let mut out = format!("{:?}", alloc_id);
                    if ptr_offset > 0 {
                        write!(out, "+{:#x}", ptr_offset).unwrap();
                    } else if ptr_offset < 0 {
                        write!(out, "-{:#x}", ptr_offset.unsigned_abs()).unwrap();
                    }
                    out
                });
                diag.arg("inbounds_size", inbounds_size);
                diag.arg("inbounds_size_is_neg", inbounds_size < 0);
                diag.arg("inbounds_size_abs", inbounds_size.unsigned_abs());
                diag.arg("ptr_offset", ptr_offset);
                diag.arg("ptr_offset_is_neg", ptr_offset < 0);
                diag.arg("ptr_offset_abs", ptr_offset.unsigned_abs());
                diag.arg(
                    "alloc_size_minus_ptr_offset",
                    alloc_size.bytes().saturating_sub(ptr_offset as u64),
                );
                diag.arg("operation", format!("{:?}", msg));
            }
            DanglingIntPointer { addr, inbounds_size, msg } => {
                if addr != 0 {
                    diag.arg(
                        "pointer",
                        Pointer::<Option<CtfeProvenance>>::without_provenance(addr).to_string(),
                    );
                }

                diag.arg("inbounds_size", inbounds_size);
                diag.arg("inbounds_size_is_neg", inbounds_size < 0);
                diag.arg("inbounds_size_abs", inbounds_size.unsigned_abs());
                diag.arg("operation", format!("{:?}", msg));
            }
            AlignmentCheckFailed(Misalignment { required, has }, msg) => {
                diag.arg("required", required.bytes());
                diag.arg("has", has.bytes());
                diag.arg("msg", format!("{msg:?}"));
            }
            WriteToReadOnly(alloc)
            | DerefFunctionPointer(alloc)
            | DerefVTablePointer(alloc)
            | DerefVaListPointer(alloc)
            | DerefTypeIdPointer(alloc) => {
                diag.arg("allocation", alloc);
            }
            InvalidBool(b) => {
                diag.arg("value", format!("{b:02x}"));
            }
            InvalidChar(c) => {
                diag.arg("value", format!("{c:08x}"));
            }
            InvalidTag(tag) => {
                diag.arg("tag", format!("{tag:x}"));
            }
            InvalidStr(err) => {
                diag.arg("err", format!("{err}"));
            }
            InvalidUninitBytes(Some((alloc, info))) => {
                diag.arg("alloc", alloc);
                diag.arg("access", info.access);
                diag.arg("uninit", info.bad);
            }
            ScalarSizeMismatch(info) => {
                diag.arg("target_size", info.target_size);
                diag.arg("data_size", info.data_size);
            }
            InvalidNichedEnumVariantWritten { enum_ty } => {
                diag.arg("ty", enum_ty);
            }
            AbiMismatchArgument { arg_idx, caller_ty, callee_ty } => {
                diag.arg("arg_idx", arg_idx + 1); // adjust for 1-indexed lists in output
                diag.arg("caller_ty", caller_ty);
                diag.arg("callee_ty", callee_ty);
            }
            AbiMismatchReturn { caller_ty, callee_ty } => {
                diag.arg("caller_ty", caller_ty);
                diag.arg("callee_ty", callee_ty);
            }
            CVariadicMismatch { caller_is_c_variadic, callee_is_c_variadic } => {
                diag.arg("caller_is_c_variadic", caller_is_c_variadic);
                diag.arg("callee_is_c_variadic", callee_is_c_variadic);
            }
            CVariadicFixedCountMismatch { caller, callee } => {
                diag.arg("caller", caller);
                diag.arg("callee", callee);
            }
        }
    }
}

impl<'tcx> ReportErrorExt for ValidationErrorInfo<'tcx> {
    fn diagnostic_message(&self) -> DiagMessage {
        use rustc_middle::mir::interpret::ValidationErrorKind::*;

        match self.kind {
            PtrToUninhabited { ptr_kind: PointerKind::Box, .. } => {
                msg!("{$front_matter}: encountered a box pointing to uninhabited type {$ty}")
            }
            PtrToUninhabited { ptr_kind: PointerKind::Ref(_), .. } => {
                msg!("{$front_matter}: encountered a reference pointing to uninhabited type {$ty}")
            }

            PointerAsInt { .. } => {
                msg!("{$front_matter}: encountered a pointer, but {$expected}")
            }
            PartialPointer => {
                msg!("{$front_matter}: encountered a partial pointer or a mix of pointers")
            }
            MutableRefToImmutable => {
                msg!(
                    "{$front_matter}: encountered mutable reference or box pointing to read-only memory"
                )
            }
            NullFnPtr { .. } => {
                msg!(
                    "{$front_matter}: encountered a {$maybe ->
                        [true] maybe-null
                        *[false] null
                    } function pointer"
                )
            }
            NeverVal => {
                msg!("{$front_matter}: encountered a value of the never type `!`")
            }
            NonnullPtrMaybeNull { .. } => {
                msg!(
                    "{$front_matter}: encountered a maybe-null pointer, but expected something that is definitely non-zero"
                )
            }
            PtrOutOfRange { .. } => {
                msg!(
                    "{$front_matter}: encountered a pointer with unknown absolute address, but expected something that is definitely {$in_range}"
                )
            }
            OutOfRange { .. } => {
                msg!("{$front_matter}: encountered {$value}, but expected something {$in_range}")
            }
            UnsafeCellInImmutable => {
                msg!("{$front_matter}: encountered `UnsafeCell` in read-only memory")
            }
            UninhabitedVal { .. } => {
                msg!("{$front_matter}: encountered a value of uninhabited type `{$ty}`")
            }
            InvalidEnumTag { .. } => {
                msg!("{$front_matter}: encountered {$value}, but expected a valid enum tag")
            }
            UninhabitedEnumVariant => {
                msg!("{$front_matter}: encountered an uninhabited enum variant")
            }
            Uninit { .. } => {
                msg!("{$front_matter}: encountered uninitialized memory, but {$expected}")
            }
            InvalidVTablePtr { .. } => {
                msg!("{$front_matter}: encountered {$value}, but expected a vtable pointer")
            }
            InvalidMetaWrongTrait { .. } => {
                msg!(
                    "{$front_matter}: wrong trait in wide pointer vtable: expected `{$expected_dyn_type}`, but encountered `{$vtable_dyn_type}`"
                )
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Box } => {
                msg!(
                    "{$front_matter}: encountered invalid box metadata: slice is bigger than largest supported object"
                )
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Ref(_) } => {
                msg!(
                    "{$front_matter}: encountered invalid reference metadata: slice is bigger than largest supported object"
                )
            }

            InvalidMetaTooLarge { ptr_kind: PointerKind::Box } => {
                msg!(
                    "{$front_matter}: encountered invalid box metadata: total size is bigger than largest supported object"
                )
            }
            InvalidMetaTooLarge { ptr_kind: PointerKind::Ref(_) } => {
                msg!(
                    "{$front_matter}: encountered invalid reference metadata: total size is bigger than largest supported object"
                )
            }
            UnalignedPtr { ptr_kind: PointerKind::Ref(_), .. } => {
                msg!(
                    "{$front_matter}: encountered an unaligned reference (required {$required_bytes} byte alignment but found {$found_bytes})"
                )
            }
            UnalignedPtr { ptr_kind: PointerKind::Box, .. } => {
                msg!(
                    "{$front_matter}: encountered an unaligned box (required {$required_bytes} byte alignment but found {$found_bytes})"
                )
            }

            NullPtr { ptr_kind: PointerKind::Box, .. } => {
                msg!(
                    "{$front_matter}: encountered a {$maybe ->
                        [true] maybe-null
                        *[false] null
                    } box"
                )
            }
            NullPtr { ptr_kind: PointerKind::Ref(_), .. } => {
                msg!(
                    "{$front_matter}: encountered a {$maybe ->
                        [true] maybe-null
                        *[false] null
                    } reference"
                )
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Box, .. } => {
                msg!("{$front_matter}: encountered a dangling box ({$pointer} has no provenance)")
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Ref(_), .. } => {
                msg!(
                    "{$front_matter}: encountered a dangling reference ({$pointer} has no provenance)"
                )
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Box } => {
                msg!(
                    "{$front_matter}: encountered a dangling box (going beyond the bounds of its allocation)"
                )
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Ref(_) } => {
                msg!(
                    "{$front_matter}: encountered a dangling reference (going beyond the bounds of its allocation)"
                )
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Box } => {
                msg!("{$front_matter}: encountered a dangling box (use-after-free)")
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Ref(_) } => {
                msg!("{$front_matter}: encountered a dangling reference (use-after-free)")
            }
            InvalidBool { .. } => {
                msg!("{$front_matter}: encountered {$value}, but expected a boolean")
            }
            InvalidChar { .. } => {
                msg!(
                    "{$front_matter}: encountered {$value}, but expected a valid unicode scalar value (in `0..=0x10FFFF` but not in `0xD800..=0xDFFF`)"
                )
            }
            InvalidFnPtr { .. } => {
                msg!("{$front_matter}: encountered {$value}, but expected a function pointer")
            }
        }
    }

    fn add_args<G: EmissionGuarantee>(self, err: &mut Diag<'_, G>) {
        use rustc_errors::msg;
        use rustc_middle::mir::interpret::ValidationErrorKind::*;

        if let PointerAsInt { .. } | PartialPointer = self.kind {
            err.help(msg!("this code performed an operation that depends on the underlying bytes representing a pointer"));
            err.help(msg!("the absolute address of a pointer is not known at compile-time, so such operations are not supported"));
        }

        let message = if let Some(path) = self.path {
            err.dcx.eagerly_translate_to_string(
                msg!("constructing invalid value at {$path}"),
                [("path".into(), DiagArgValue::Str(path.into()))].iter().map(|(a, b)| (a, b)),
            )
        } else {
            err.dcx.eagerly_translate_to_string(msg!("constructing invalid value"), [].into_iter())
        };

        err.arg("front_matter", message);

        fn add_range_arg<G: EmissionGuarantee>(
            r: WrappingRange,
            max_hi: u128,
            err: &mut Diag<'_, G>,
        ) {
            let WrappingRange { start: lo, end: hi } = r;
            assert!(hi <= max_hi);
            let msg = if lo > hi {
                msg!("less or equal to {$hi}, or greater or equal to {$lo}")
            } else if lo == hi {
                msg!("equal to {$lo}")
            } else if lo == 0 {
                assert!(hi < max_hi, "should not be printing if the range covers everything");
                msg!("less or equal to {$hi}")
            } else if hi == max_hi {
                assert!(lo > 0, "should not be printing if the range covers everything");
                msg!("greater or equal to {$lo}")
            } else {
                msg!("in the range {$lo}..={$hi}")
            };

            let args = [
                ("lo".into(), DiagArgValue::Str(lo.to_string().into())),
                ("hi".into(), DiagArgValue::Str(hi.to_string().into())),
            ];
            let args = args.iter().map(|(a, b)| (a, b));
            let message = err.dcx.eagerly_translate_to_string(msg, args);
            err.arg("in_range", message);
        }

        match self.kind {
            PtrToUninhabited { ty, .. } | UninhabitedVal { ty } => {
                err.arg("ty", ty);
            }
            PointerAsInt { expected } | Uninit { expected } => {
                let msg = match expected {
                    ExpectedKind::Reference => msg!("expected a reference"),
                    ExpectedKind::Box => msg!("expected a box"),
                    ExpectedKind::RawPtr => msg!("expected a raw pointer"),
                    ExpectedKind::InitScalar => msg!("expected initialized scalar value"),
                    ExpectedKind::Bool => msg!("expected a boolean"),
                    ExpectedKind::Char => msg!("expected a unicode scalar value"),
                    ExpectedKind::Float => msg!("expected a floating point number"),
                    ExpectedKind::Int => msg!("expected an integer"),
                    ExpectedKind::FnPtr => msg!("expected a function pointer"),
                    ExpectedKind::EnumTag => msg!("expected a valid enum tag"),
                    ExpectedKind::Str => msg!("expected a string"),
                };
                let msg = err.dcx.eagerly_translate_to_string(msg, [].into_iter());
                err.arg("expected", msg);
            }
            InvalidEnumTag { value }
            | InvalidVTablePtr { value }
            | InvalidBool { value }
            | InvalidChar { value }
            | InvalidFnPtr { value } => {
                err.arg("value", value);
            }
            PtrOutOfRange { range, max_value } => add_range_arg(range, max_value, err),
            OutOfRange { range, max_value, value } => {
                err.arg("value", value);
                add_range_arg(range, max_value, err);
            }
            UnalignedPtr { required_bytes, found_bytes, .. } => {
                err.arg("required_bytes", required_bytes);
                err.arg("found_bytes", found_bytes);
            }
            DanglingPtrNoProvenance { pointer, .. } => {
                err.arg("pointer", pointer);
            }
            InvalidMetaWrongTrait { vtable_dyn_type, expected_dyn_type } => {
                err.arg("vtable_dyn_type", vtable_dyn_type.to_string());
                err.arg("expected_dyn_type", expected_dyn_type.to_string());
            }
            NullPtr { maybe, .. } | NullFnPtr { maybe } => {
                err.arg("maybe", maybe);
            }
            MutableRefToImmutable
            | NonnullPtrMaybeNull
            | NeverVal
            | UnsafeCellInImmutable
            | InvalidMetaSliceTooLarge { .. }
            | InvalidMetaTooLarge { .. }
            | DanglingPtrUseAfterFree { .. }
            | DanglingPtrOutOfBounds { .. }
            | UninhabitedEnumVariant
            | PartialPointer => {}
        }
    }
}

impl ReportErrorExt for UnsupportedOpInfo {
    fn diagnostic_message(&self) -> DiagMessage {
        match self {
            UnsupportedOpInfo::Unsupported(s) => s.clone().into(),
            UnsupportedOpInfo::ExternTypeField => {
                "`extern type` field does not have a known offset".into()
            }
            UnsupportedOpInfo::UnsizedLocal => "unsized locals are not supported".into(),
            UnsupportedOpInfo::ReadPartialPointer(_) => {
                msg!("unable to read parts of a pointer from memory at {$ptr}")
            }
            UnsupportedOpInfo::ReadPointerAsInt(_) => "unable to turn pointer into integer".into(),
            UnsupportedOpInfo::ThreadLocalStatic(_) => {
                msg!("cannot access thread local static `{$did}`")
            }
            UnsupportedOpInfo::ExternStatic(_) => {
                msg!("cannot access extern static `{$did}`")
            }
        }
        .into()
    }

    fn add_args<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        use UnsupportedOpInfo::*;

        if let ReadPointerAsInt(_) | ReadPartialPointer(_) = self {
            diag.help("this code performed an operation that depends on the underlying bytes representing a pointer");
            diag.help("the absolute address of a pointer is not known at compile-time, so such operations are not supported");
        }
        match self {
            // `ReadPointerAsInt(Some(info))` is never printed anyway, it only serves as an error to
            // be further processed by validity checking which then turns it into something nice to
            // print. So it's not worth the effort of having diagnostics that can print the `info`.
            UnsizedLocal
            | UnsupportedOpInfo::ExternTypeField
            | Unsupported(_)
            | ReadPointerAsInt(_) => {}
            ReadPartialPointer(ptr) => {
                diag.arg("ptr", ptr);
            }
            ThreadLocalStatic(did) | ExternStatic(did) => rustc_middle::ty::tls::with(|tcx| {
                diag.arg("did", tcx.def_path_str(did));
            }),
        }
    }
}

impl<'tcx> ReportErrorExt for InterpErrorKind<'tcx> {
    fn diagnostic_message(&self) -> DiagMessage {
        match self {
            InterpErrorKind::UndefinedBehavior(ub) => ub.diagnostic_message(),
            InterpErrorKind::Unsupported(e) => e.diagnostic_message(),
            InterpErrorKind::InvalidProgram(e) => e.diagnostic_message(),
            InterpErrorKind::ResourceExhaustion(e) => e.diagnostic_message(),
            InterpErrorKind::MachineStop(e) => e.diagnostic_message(),
        }
    }
    fn add_args<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            InterpErrorKind::UndefinedBehavior(ub) => ub.add_args(diag),
            InterpErrorKind::Unsupported(e) => e.add_args(diag),
            InterpErrorKind::InvalidProgram(e) => e.add_args(diag),
            InterpErrorKind::ResourceExhaustion(e) => e.add_args(diag),
            InterpErrorKind::MachineStop(e) => e.add_args(&mut |name, value| {
                diag.arg(name, value);
            }),
        }
    }
}

impl<'tcx> ReportErrorExt for InvalidProgramInfo<'tcx> {
    fn diagnostic_message(&self) -> DiagMessage {
        match self {
            InvalidProgramInfo::TooGeneric => "encountered overly generic constant".into(),
            InvalidProgramInfo::AlreadyReported(_) => {
                "an error has already been reported elsewhere (this should not usually be printed)"
                    .into()
            }
            InvalidProgramInfo::Layout(e) => e.diagnostic_message(),
        }
    }
    fn add_args<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        match self {
            InvalidProgramInfo::TooGeneric | InvalidProgramInfo::AlreadyReported(_) => {}
            InvalidProgramInfo::Layout(e) => {
                // The level doesn't matter, `dummy_diag` is consumed without it being used.
                let dummy_level = Level::Bug;
                let dummy_diag: Diag<'_, ()> = e.into_diagnostic().into_diag(diag.dcx, dummy_level);
                for (name, val) in dummy_diag.args.iter() {
                    diag.arg(name.clone(), val.clone());
                }
                dummy_diag.cancel();
            }
        }
    }
}

impl ReportErrorExt for ResourceExhaustionInfo {
    fn diagnostic_message(&self) -> DiagMessage {
        match self {
            ResourceExhaustionInfo::StackFrameLimitReached => {
                "reached the configured maximum number of stack frames"
            }
            ResourceExhaustionInfo::MemoryExhausted => {
                "tried to allocate more memory than available to compiler"
            }
            ResourceExhaustionInfo::AddressSpaceFull => {
                "there are no more free addresses in the address space"
            }
            ResourceExhaustionInfo::Interrupted => "compilation was interrupted",
        }
        .into()
    }
    fn add_args<G: EmissionGuarantee>(self, _: &mut Diag<'_, G>) {}
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
