use std::backtrace::Backtrace;
use std::borrow::Cow;
use std::fmt;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use rustc_ast as ast;
use rustc_ast_pretty::pprust;
use rustc_span::edition::Edition;

use crate::{DiagArgValue, IntoDiagArg};

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
            impl $crate::IntoDiagArg for $ty {
                fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> $crate::DiagArgValue {
                    self.to_string().into_diag_arg(path)
                }
            }
        )+
    }
}

macro_rules! into_diag_arg_for_number {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl $crate::IntoDiagArg for $ty {
                fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> $crate::DiagArgValue {
                    // Convert to a string if it won't fit into `Number`.
                    #[allow(irrefutable_let_patterns)]
                    if let Ok(n) = TryInto::<i32>::try_into(self) {
                        $crate::DiagArgValue::Number(n)
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
    Edition,
    rustc_span::Ident,
    rustc_span::MacroRulesNormalizedIdent,
    ParseIntError,
    ExitStatus,
);

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

impl IntoDiagArg for rustc_span::Symbol {
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

impl IntoDiagArg for Backtrace {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
    }
}

impl IntoDiagArg for ast::util::parser::ExprPrecedence {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Number(self as i32)
    }
}

impl IntoDiagArg for ast::FloatTy {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::Borrowed(self.name_str()))
    }
}
