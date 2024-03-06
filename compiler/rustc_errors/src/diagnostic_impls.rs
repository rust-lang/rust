use crate::diagnostic::DiagLocation;
use crate::{fluent_generated as fluent, AddToDiagnostic};
use crate::{
    Diag, DiagArgValue, DiagCtxt, EmissionGuarantee, ErrCode, IntoDiagnostic, IntoDiagnosticArg,
    Level, SubdiagMessageOp,
};
use rustc_ast as ast;
use rustc_ast_pretty::pprust;
use rustc_hir as hir;
use rustc_span::edition::Edition;
use rustc_span::symbol::{Ident, MacroRulesNormalizedIdent, Symbol};
use rustc_span::Span;
use rustc_target::abi::TargetDataLayoutErrors;
use rustc_target::spec::{PanicStrategy, SplitDebuginfo, StackProtector, TargetTriple};
use rustc_type_ir as type_ir;
use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::fmt;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

pub struct DiagArgFromDisplay<'a>(pub &'a dyn fmt::Display);

impl IntoDiagnosticArg for DiagArgFromDisplay<'_> {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        self.0.to_string().into_diagnostic_arg()
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

impl<'a, T: Clone + IntoDiagnosticArg> IntoDiagnosticArg for &'a T {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        self.clone().into_diagnostic_arg()
    }
}

macro_rules! into_diagnostic_arg_using_display {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl IntoDiagnosticArg for $ty {
                fn into_diagnostic_arg(self) -> DiagArgValue {
                    self.to_string().into_diagnostic_arg()
                }
            }
        )+
    }
}

macro_rules! into_diagnostic_arg_for_number {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl IntoDiagnosticArg for $ty {
                fn into_diagnostic_arg(self) -> DiagArgValue {
                    // Convert to a string if it won't fit into `Number`.
                    if let Ok(n) = TryInto::<i32>::try_into(self) {
                        DiagArgValue::Number(n)
                    } else {
                        self.to_string().into_diagnostic_arg()
                    }
                }
            }
        )+
    }
}

into_diagnostic_arg_using_display!(
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
    &TargetTriple,
    SplitDebuginfo,
    ExitStatus,
    ErrCode,
);

into_diagnostic_arg_for_number!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, isize, usize);

impl IntoDiagnosticArg for bool {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        if self {
            DiagArgValue::Str(Cow::Borrowed("true"))
        } else {
            DiagArgValue::Str(Cow::Borrowed("false"))
        }
    }
}

impl IntoDiagnosticArg for char {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(format!("{self:?}")))
    }
}

impl IntoDiagnosticArg for Vec<char> {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::StrListSepByAnd(
            self.into_iter().map(|c| Cow::Owned(format!("{c:?}"))).collect(),
        )
    }
}

impl IntoDiagnosticArg for Symbol {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        self.to_ident_string().into_diagnostic_arg()
    }
}

impl<'a> IntoDiagnosticArg for &'a str {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        self.to_string().into_diagnostic_arg()
    }
}

impl IntoDiagnosticArg for String {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self))
    }
}

impl<'a> IntoDiagnosticArg for Cow<'a, str> {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.into_owned()))
    }
}

impl<'a> IntoDiagnosticArg for &'a Path {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagnosticArg for PathBuf {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagnosticArg for PanicStrategy {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.desc().to_string()))
    }
}

impl IntoDiagnosticArg for hir::ConstContext {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(match self {
            hir::ConstContext::ConstFn => "const_fn",
            hir::ConstContext::Static(_) => "static",
            hir::ConstContext::Const { .. } => "const",
        }))
    }
}

impl IntoDiagnosticArg for ast::Expr {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(pprust::expr_to_string(&self)))
    }
}

impl IntoDiagnosticArg for ast::Path {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(pprust::path_to_string(&self)))
    }
}

impl IntoDiagnosticArg for ast::token::Token {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(pprust::token_to_string(&self))
    }
}

impl IntoDiagnosticArg for ast::token::TokenKind {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(pprust::token_kind_to_string(&self))
    }
}

impl IntoDiagnosticArg for type_ir::FloatTy {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.name_str()))
    }
}

impl IntoDiagnosticArg for std::ffi::CString {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.to_string_lossy().into_owned()))
    }
}

impl IntoDiagnosticArg for rustc_data_structures::small_c_str::SmallCStr {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Owned(self.to_string_lossy().into_owned()))
    }
}

impl IntoDiagnosticArg for ast::Visibility {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        let s = pprust::vis_to_string(&self);
        let s = s.trim_end().to_string();
        DiagArgValue::Str(Cow::Owned(s))
    }
}

impl IntoDiagnosticArg for rustc_lint_defs::Level {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.to_cmd_flag()))
    }
}

#[derive(Clone)]
pub struct DiagSymbolList(Vec<Symbol>);

impl From<Vec<Symbol>> for DiagSymbolList {
    fn from(v: Vec<Symbol>) -> Self {
        DiagSymbolList(v)
    }
}

impl IntoDiagnosticArg for DiagSymbolList {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::StrListSepByAnd(
            self.0.into_iter().map(|sym| Cow::Owned(format!("`{sym}`"))).collect(),
        )
    }
}

impl<Id> IntoDiagnosticArg for hir::def::Res<Id> {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.descr()))
    }
}

impl<G: EmissionGuarantee> IntoDiagnostic<'_, G> for TargetDataLayoutErrors<'_> {
    fn into_diagnostic(self, dcx: &DiagCtxt, level: Level) -> Diag<'_, G> {
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
impl AddToDiagnostic for SingleLabelManySpans {
    fn add_to_diagnostic_with<G: EmissionGuarantee, F: SubdiagMessageOp<G>>(
        self,
        diag: &mut Diag<'_, G>,
        _: F,
    ) {
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

impl IntoDiagnosticArg for DiagLocation {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
    }
}

impl IntoDiagnosticArg for Backtrace {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
    }
}

impl IntoDiagnosticArg for Level {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
    }
}

#[derive(Subdiagnostic)]
#[suggestion(errors_indicate_anonymous_lifetime, code = "{suggestion}", style = "verbose")]
pub struct IndicateAnonymousLifetime {
    #[primary_span]
    pub span: Span,
    pub count: usize,
    pub suggestion: String,
}

impl IntoDiagnosticArg for type_ir::ClosureKind {
    fn into_diagnostic_arg(self) -> DiagArgValue {
        DiagArgValue::Str(self.as_str().into())
    }
}
