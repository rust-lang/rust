use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::fmt;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use rustc_abi::TargetDataLayoutErrors;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_ast_pretty::pprust;
use rustc_attr_data_structures::RustcVersion;
use rustc_macros::Subdiagnostic;
use rustc_span::edition::Edition;
use rustc_span::{Ident, MacroRulesNormalizedIdent, Span, Symbol};
use rustc_target::spec::{PanicStrategy, SplitDebuginfo, StackProtector, TargetTuple};
use rustc_type_ir::{ClosureKind, FloatTy};
use {rustc_ast as ast, rustc_hir as hir};

use crate::diagnostic::DiagLocation;
use crate::{
    Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, ErrCode, IntoDiagArg, Level,
    Subdiagnostic, fluent_generated as fluent,
};

pub struct DiagArgFromDisplay<'a>(pub &'a dyn fmt::Display);

impl IntoDiagArg for DiagArgFromDisplay<'_> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.0.to_string().into_diag_arg(path)
    }
}

impl<'a> From<&'a dyn fmt::Display> for DiagArgFromDisplay<'a> {
    fn from(t: &'a dyn fmt::Display) -> Self {
        DiagArgFromDisplay(t)
    }
}

impl<'a, T: fmt::Display> From<&'a T> for DiagArgFromDisplay<'a> {
    fn from(t: &'a T) -> Self {
        DiagArgFromDisplay(t)
    }
}

impl<'a, T: Clone + IntoDiagArg> IntoDiagArg for &'a T {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.clone().into_diag_arg(path)
    }
}

#[macro_export]
macro_rules! into_diag_arg_using_display {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl IntoDiagArg for $ty {
                fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
                    self.to_string().into_diag_arg(path)
                }
            }
        )+
    }
}

macro_rules! into_diag_arg_for_number {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl IntoDiagArg for $ty {
                fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
                    // Convert to a string if it won't fit into `Number`.
                    #[allow(irrefutable_let_patterns)]
                    if let Ok(n) = TryInto::<i32>::try_into(self) {
                        DiagArgValue::Number(n)
                    } else {
                        self.to_string().into_diag_arg(path)
                    }
                }
            }
        )+
    }
}

into_diag_arg_using_display!(
    ast::ParamKindOrd,
    std::io::Error,
    Box<dyn std::error::Error>,
    std::num::NonZero<u32>,
    hir::Target,
    Edition,
    Ident,
    MacroRulesNormalizedIdent,
    ParseIntError,
    StackProtector,
    &TargetTuple,
    SplitDebuginfo,
    ExitStatus,
    ErrCode,
    rustc_abi::ExternAbi,
);

impl IntoDiagArg for RustcVersion {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.to_string()))
    }
}

impl<I: rustc_type_ir::Interner> IntoDiagArg for rustc_type_ir::TraitRef<I> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(path)
    }
}

impl<I: rustc_type_ir::Interner> IntoDiagArg for rustc_type_ir::ExistentialTraitRef<I> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(path)
    }
}

impl<I: rustc_type_ir::Interner> IntoDiagArg for rustc_type_ir::UnevaluatedConst<I> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        format!("{self:?}").into_diag_arg(path)
    }
}

impl<I: rustc_type_ir::Interner> IntoDiagArg for rustc_type_ir::FnSig<I> {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        format!("{self:?}").into_diag_arg(path)
    }
}

impl<I: rustc_type_ir::Interner, T> IntoDiagArg for rustc_type_ir::Binder<I, T>
where
    T: IntoDiagArg,
{
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.skip_binder().into_diag_arg(path)
    }
}

into_diag_arg_for_number!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize);

impl IntoDiagArg for bool {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        if self {
            DiagArgValue::Str(Cow::Borrowed("true"))
        } else {
            DiagArgValue::Str(Cow::Borrowed("false"))
        }
    }
}

impl IntoDiagArg for char {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(format!("{self:?}")))
    }
}

impl IntoDiagArg for Vec<char> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::StrListSepByAnd(
            self.into_iter().map(|c| Cow::Owned(format!("{c:?}"))).collect(),
        )
    }
}

impl IntoDiagArg for Symbol {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_ident_string().into_diag_arg(path)
    }
}

impl<'a> IntoDiagArg for &'a str {
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self.to_string().into_diag_arg(path)
    }
}

impl IntoDiagArg for String {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self))
    }
}

impl<'a> IntoDiagArg for Cow<'a, str> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.into_owned()))
    }
}

impl<'a> IntoDiagArg for &'a Path {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagArg for PathBuf {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagArg for PanicStrategy {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.desc().to_string()))
    }
}

impl IntoDiagArg for hir::ConstContext {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(match self {
            hir::ConstContext::ConstFn => "const_fn",
            hir::ConstContext::Static(_) => "static",
            hir::ConstContext::Const { .. } => "const",
        }))
    }
}

impl IntoDiagArg for ast::Expr {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(pprust::expr_to_string(&self)))
    }
}

impl IntoDiagArg for ast::Path {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(pprust::path_to_string(&self)))
    }
}

impl IntoDiagArg for ast::token::Token {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(pprust::token_to_string(&self))
    }
}

impl IntoDiagArg for ast::token::TokenKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(pprust::token_kind_to_string(&self))
    }
}

impl IntoDiagArg for FloatTy {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.name_str()))
    }
}

impl IntoDiagArg for std::ffi::CString {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.to_string_lossy().into_owned()))
    }
}

impl IntoDiagArg for rustc_data_structures::small_c_str::SmallCStr {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.to_string_lossy().into_owned()))
    }
}

impl IntoDiagArg for ast::Visibility {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        let s = pprust::vis_to_string(&self);
        let s = s.trim_end().to_string();
        DiagArgValue::Str(Cow::Owned(s))
    }
}

impl IntoDiagArg for rustc_lint_defs::Level {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.to_cmd_flag()))
    }
}

impl<Id> IntoDiagArg for hir::def::Res<Id> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.descr()))
    }
}

impl IntoDiagArg for DiagLocation {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
    }
}

impl IntoDiagArg for Backtrace {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
    }
}

impl IntoDiagArg for Level {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
    }
}

impl IntoDiagArg for ClosureKind {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(self.as_str().into())
    }
}

impl IntoDiagArg for hir::def::Namespace {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.descr()))
    }
}

impl IntoDiagArg for ExprPrecedence {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Number(self as i32)
    }
}

#[derive(Clone)]
pub struct DiagSymbolList<S = Symbol>(Vec<S>);

impl<S> From<Vec<S>> for DiagSymbolList<S> {
    fn from(v: Vec<S>) -> Self {
        DiagSymbolList(v)
    }
}

impl<S> FromIterator<S> for DiagSymbolList<S> {
    fn from_iter<T: IntoIterator<Item = S>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl<S: std::fmt::Display> IntoDiagArg for DiagSymbolList<S> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::StrListSepByAnd(
            self.0.into_iter().map(|sym| Cow::Owned(format!("`{sym}`"))).collect(),
        )
    }
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for TargetDataLayoutErrors<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        match self {
            TargetDataLayoutErrors::InvalidAddressSpace { addr_space, err, cause } => {
                Diag::new(dcx, level, fluent::errors_target_invalid_address_space)
                    .with_arg("addr_space", addr_space)
                    .with_arg("cause", cause)
                    .with_arg("err", err)
            }
            TargetDataLayoutErrors::InvalidBits { kind, bit, cause, err } => {
                Diag::new(dcx, level, fluent::errors_target_invalid_bits)
                    .with_arg("kind", kind)
                    .with_arg("bit", bit)
                    .with_arg("cause", cause)
                    .with_arg("err", err)
            }
            TargetDataLayoutErrors::MissingAlignment { cause } => {
                Diag::new(dcx, level, fluent::errors_target_missing_alignment)
                    .with_arg("cause", cause)
            }
            TargetDataLayoutErrors::InvalidAlignment { cause, err } => {
                Diag::new(dcx, level, fluent::errors_target_invalid_alignment)
                    .with_arg("cause", cause)
                    .with_arg("err_kind", err.diag_ident())
                    .with_arg("align", err.align())
            }
            TargetDataLayoutErrors::InconsistentTargetArchitecture { dl, target } => {
                Diag::new(dcx, level, fluent::errors_target_inconsistent_architecture)
                    .with_arg("dl", dl)
                    .with_arg("target", target)
            }
            TargetDataLayoutErrors::InconsistentTargetPointerWidth { pointer_size, target } => {
                Diag::new(dcx, level, fluent::errors_target_inconsistent_pointer_width)
                    .with_arg("pointer_size", pointer_size)
                    .with_arg("target", target)
            }
            TargetDataLayoutErrors::InvalidBitsSize { err } => {
                Diag::new(dcx, level, fluent::errors_target_invalid_bits_size).with_arg("err", err)
            }
        }
    }
}

/// Utility struct used to apply a single label while highlighting multiple spans
pub struct SingleLabelManySpans {
    pub spans: Vec<Span>,
    pub label: &'static str,
}
impl Subdiagnostic for SingleLabelManySpans {
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>) {
        diag.span_labels(self.spans, self.label);
    }
}

#[derive(Subdiagnostic)]
#[label(errors_expected_lifetime_parameter)]
pub struct ExpectedLifetimeParameter {
    #[primary_span]
    pub span: Span,
    pub count: usize,
}

#[derive(Subdiagnostic)]
#[suggestion(errors_indicate_anonymous_lifetime, code = "{suggestion}", style = "verbose")]
pub struct IndicateAnonymousLifetime {
    #[primary_span]
    pub span: Span,
    pub count: usize,
    pub suggestion: String,
}

#[derive(Subdiagnostic)]
pub struct ElidedLifetimeInPathSubdiag {
    #[subdiagnostic]
    pub expected: ExpectedLifetimeParameter,
    #[subdiagnostic]
    pub indicate: Option<IndicateAnonymousLifetime>,
}
