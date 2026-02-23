use std::borrow::Cow;

use rustc_abi::TargetDataLayoutErrors;
use rustc_error_messages::{DiagArgValue, IntoDiagArg};
use rustc_macros::Subdiagnostic;
use rustc_span::{Span, Symbol};

use crate::diagnostic::DiagLocation;
use crate::{Diag, DiagCtxtHandle, Diagnostic, EmissionGuarantee, Level, Subdiagnostic, msg};

impl IntoDiagArg for DiagLocation {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        DiagArgValue::Str(Cow::from(self.to_string()))
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
                Diag::new(dcx, level, msg!("invalid address space `{$addr_space}` for `{$cause}` in \"data-layout\": {$err}"))
                    .with_arg("addr_space", addr_space)
                    .with_arg("cause", cause)
                    .with_arg("err", err)
            }
            TargetDataLayoutErrors::InvalidBits { kind, bit, cause, err } => {
                Diag::new(dcx, level, msg!("invalid {$kind} `{$bit}` for `{$cause}` in \"data-layout\": {$err}"))
                    .with_arg("kind", kind)
                    .with_arg("bit", bit)
                    .with_arg("cause", cause)
                    .with_arg("err", err)
            }
            TargetDataLayoutErrors::MissingAlignment { cause } => {
                Diag::new(dcx, level, msg!("missing alignment for `{$cause}` in \"data-layout\""))
                    .with_arg("cause", cause)
            }
            TargetDataLayoutErrors::InvalidAlignment { cause, err } => {
                Diag::new(dcx, level, msg!(
                    "invalid alignment for `{$cause}` in \"data-layout\": `{$align}` is {$err_kind ->
                        [not_power_of_two] not a power of 2
                        [too_large] too large
                        *[other] {\"\"}
                    }"
                ))
                .with_arg("cause", cause)
                .with_arg("err_kind", err.diag_ident())
                .with_arg("align", err.align())
            }
            TargetDataLayoutErrors::InconsistentTargetArchitecture { dl, target } => {
                Diag::new(dcx, level, msg!(
                    "inconsistent target specification: \"data-layout\" claims architecture is {$dl}-endian, while \"target-endian\" is `{$target}`"
                ))
                .with_arg("dl", dl).with_arg("target", target)
            }
            TargetDataLayoutErrors::InconsistentTargetPointerWidth { pointer_size, target } => {
                Diag::new(dcx, level, msg!(
                    "inconsistent target specification: \"data-layout\" claims pointers are {$pointer_size}-bit, while \"target-pointer-width\" is `{$target}`"
                )).with_arg("pointer_size", pointer_size).with_arg("target", target)
            }
            TargetDataLayoutErrors::InvalidBitsSize { err } => {
                Diag::new(dcx, level, msg!("{$err}")).with_arg("err", err)
            }
            TargetDataLayoutErrors::UnknownPointerSpecification { err } => {
                Diag::new(dcx, level, msg!("unknown pointer specification `{$err}` in datalayout string"))
                    .with_arg("err", err)
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
#[label(
    "expected lifetime {$count ->
        [1] parameter
        *[other] parameters
    }"
)]
pub struct ExpectedLifetimeParameter {
    #[primary_span]
    pub span: Span,
    pub count: usize,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "indicate the anonymous {$count ->
        [1] lifetime
        *[other] lifetimes
    }",
    code = "{suggestion}",
    style = "verbose"
)]
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
