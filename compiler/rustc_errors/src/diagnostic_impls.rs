use crate::{
    fluent, DiagnosticArgValue, DiagnosticBuilder, Handler, IntoDiagnostic, IntoDiagnosticArg,
};
use rustc_ast as ast;
use rustc_ast_pretty::pprust;
use rustc_hir as hir;
use rustc_lint_defs::Level;
use rustc_span::edition::Edition;
use rustc_span::symbol::{Ident, MacroRulesNormalizedIdent, Symbol};
use rustc_target::abi::TargetDataLayoutErrors;
use rustc_target::spec::{PanicStrategy, SplitDebuginfo, StackProtector, TargetTriple};
use rustc_type_ir as type_ir;
use std::borrow::Cow;
use std::fmt;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

pub struct DiagnosticArgFromDisplay<'a>(pub &'a dyn fmt::Display);

impl IntoDiagnosticArg for DiagnosticArgFromDisplay<'_> {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.0.to_string().into_diagnostic_arg()
    }
}

impl<'a> From<&'a dyn fmt::Display> for DiagnosticArgFromDisplay<'a> {
    fn from(t: &'a dyn fmt::Display) -> Self {
        DiagnosticArgFromDisplay(t)
    }
}

impl<'a, T: fmt::Display> From<&'a T> for DiagnosticArgFromDisplay<'a> {
    fn from(t: &'a T) -> Self {
        DiagnosticArgFromDisplay(t)
    }
}

impl<'a, T: Clone + IntoDiagnosticArg> IntoDiagnosticArg for &'a T {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.clone().into_diagnostic_arg()
    }
}

macro_rules! into_diagnostic_arg_using_display {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl IntoDiagnosticArg for $ty {
                fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
                    self.to_string().into_diagnostic_arg()
                }
            }
        )+
    }
}

into_diagnostic_arg_using_display!(
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    std::io::Error,
    Box<dyn std::error::Error>,
    std::num::NonZeroU32,
    hir::Target,
    Edition,
    Ident,
    MacroRulesNormalizedIdent,
    ParseIntError,
    StackProtector,
    &TargetTriple,
    SplitDebuginfo,
    ExitStatus,
);

impl IntoDiagnosticArg for bool {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        if self {
            DiagnosticArgValue::Str(Cow::Borrowed("true"))
        } else {
            DiagnosticArgValue::Str(Cow::Borrowed("false"))
        }
    }
}

impl IntoDiagnosticArg for char {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(format!("{:?}", self)))
    }
}

impl IntoDiagnosticArg for Symbol {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.to_ident_string().into_diagnostic_arg()
    }
}

impl<'a> IntoDiagnosticArg for &'a str {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.to_string().into_diagnostic_arg()
    }
}

impl IntoDiagnosticArg for String {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self))
    }
}

impl<'a> IntoDiagnosticArg for Cow<'a, str> {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.into_owned()))
    }
}

impl<'a> IntoDiagnosticArg for &'a Path {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagnosticArg for PathBuf {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagnosticArg for usize {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Number(self)
    }
}

impl IntoDiagnosticArg for PanicStrategy {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.desc().to_string()))
    }
}

impl IntoDiagnosticArg for hir::ConstContext {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(match self {
            hir::ConstContext::ConstFn => "constant function",
            hir::ConstContext::Static(_) => "static",
            hir::ConstContext::Const => "constant",
        }))
    }
}

impl IntoDiagnosticArg for ast::Path {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(pprust::path_to_string(&self)))
    }
}

impl IntoDiagnosticArg for ast::token::Token {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(pprust::token_to_string(&self))
    }
}

impl IntoDiagnosticArg for ast::token::TokenKind {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(pprust::token_kind_to_string(&self))
    }
}

impl IntoDiagnosticArg for type_ir::FloatTy {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(self.name_str()))
    }
}

impl IntoDiagnosticArg for std::ffi::CString {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.to_string_lossy().into_owned()))
    }
}

impl IntoDiagnosticArg for rustc_data_structures::small_c_str::SmallCStr {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.to_string_lossy().into_owned()))
    }
}

impl IntoDiagnosticArg for Level {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(self.to_cmd_flag()))
    }
}

#[derive(Clone)]
pub struct DiagnosticSymbolList(Vec<Symbol>);

impl From<Vec<Symbol>> for DiagnosticSymbolList {
    fn from(v: Vec<Symbol>) -> Self {
        DiagnosticSymbolList(v)
    }
}

impl IntoDiagnosticArg for DiagnosticSymbolList {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::StrListSepByAnd(
            self.0.into_iter().map(|sym| Cow::Owned(format!("`{sym}`"))).collect(),
        )
    }
}

impl<Id> IntoDiagnosticArg for hir::def::Res<Id> {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(self.descr()))
    }
}

impl IntoDiagnostic<'_, !> for TargetDataLayoutErrors<'_> {
    fn into_diagnostic(self, handler: &Handler) -> DiagnosticBuilder<'_, !> {
        let mut diag;
        match self {
            TargetDataLayoutErrors::InvalidAddressSpace { addr_space, err, cause } => {
                diag = handler.struct_fatal(fluent::errors_target_invalid_address_space);
                diag.set_arg("addr_space", addr_space);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InvalidBits { kind, bit, cause, err } => {
                diag = handler.struct_fatal(fluent::errors_target_invalid_bits);
                diag.set_arg("kind", kind);
                diag.set_arg("bit", bit);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::MissingAlignment { cause } => {
                diag = handler.struct_fatal(fluent::errors_target_missing_alignment);
                diag.set_arg("cause", cause);
                diag
            }
            TargetDataLayoutErrors::InvalidAlignment { cause, err } => {
                diag = handler.struct_fatal(fluent::errors_target_invalid_alignment);
                diag.set_arg("cause", cause);
                diag.set_arg("err", err);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetArchitecture { dl, target } => {
                diag = handler.struct_fatal(fluent::errors_target_inconsistent_architecture);
                diag.set_arg("dl", dl);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InconsistentTargetPointerWidth { pointer_size, target } => {
                diag = handler.struct_fatal(fluent::errors_target_inconsistent_pointer_width);
                diag.set_arg("pointer_size", pointer_size);
                diag.set_arg("target", target);
                diag
            }
            TargetDataLayoutErrors::InvalidBitsSize { err } => {
                diag = handler.struct_fatal(fluent::errors_target_invalid_bits_size);
                diag.set_arg("err", err);
                diag
            }
        }
    }
}
