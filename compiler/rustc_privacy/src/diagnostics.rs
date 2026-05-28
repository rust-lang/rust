use rustc_errors::codes::*;
use rustc_errors::{DiagArgFromDisplay, MultiSpan};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag("{$len ->
    [1] field
    *[other] fields
} {$field_names} of {$variant_descr} `{$def_path_str}` {$len ->
    [1] is
    *[other] are
} private", code = E0451)]
pub(crate) struct FieldIsPrivate {
    #[primary_span]
    pub span: MultiSpan,
    #[label("in this type")]
    pub struct_span: Option<Span>,
    pub field_names: String,
    pub variant_descr: &'static str,
    pub def_path_str: String,
    #[subdiagnostic]
    pub labels: Vec<FieldIsPrivateLabel>,
    pub len: usize,
}

#[derive(Subdiagnostic)]
pub(crate) enum FieldIsPrivateLabel {
    #[label(
        "{$rest_len ->
            [1] field
            *[other] fields
        } {$rest_field_names} {$rest_len ->
            [1] is
            *[other] are
        } private"
    )]
    IsUpdateSyntax {
        #[primary_span]
        span: Span,
        rest_field_names: String,
        rest_len: usize,
    },
    #[label("private field")]
    Other {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("{$kind} `{$descr}` is private")]
pub(crate) struct ItemIsPrivate<'a> {
    #[primary_span]
    #[label("private {$kind}")]
    pub span: Span,
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
}

#[derive(Diagnostic)]
#[diag("{$kind} is private")]
pub(crate) struct UnnamedItemIsPrivate {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
}

#[derive(Diagnostic)]
#[diag("{$vis_descr} {$kind} `{$descr}` in public interface", code = E0446)]
pub(crate) struct InPublicInterface<'a> {
    #[primary_span]
    #[label("can't leak {$vis_descr} {$kind}")]
    pub span: Span,
    pub vis_descr: &'static str,
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
    #[label("`{$descr}` declared as {$vis_descr}")]
    pub vis_span: Span,
}

#[derive(Diagnostic)]
#[diag("{$descr}")]
pub(crate) struct ReportEffectiveVisibility {
    #[primary_span]
    pub span: Span,
    pub descr: String,
}

#[derive(Diagnostic)]
#[diag("{$kind} `{$descr}` from private dependency '{$krate}' in public interface")]
pub(crate) struct FromPrivateDependencyInPublicInterface<'a> {
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
    pub krate: Symbol,
}

#[derive(Diagnostic)]
#[diag("{$kind} `{$descr}` is reachable but cannot be named")]
pub(crate) struct UnnameableTypesLint<'a> {
    #[label(
        "reachable at visibility `{$reachable_vis}`, but can only be named at visibility `{$reexported_vis}`"
    )]
    pub span: Span,
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
    pub reachable_vis: &'a str,
    pub reexported_vis: &'a str,
}

// Used for `private_interfaces` and `private_bounds` lints.
// They will replace private-in-public errors and compatibility lints in future.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html for more details.
#[derive(Diagnostic)]
#[diag("{$ty_kind} `{$ty_descr}` is more private than the item `{$item_descr}`")]
pub(crate) struct PrivateInterfacesOrBoundsLint<'a> {
    #[label("{$item_kind} `{$item_descr}` is reachable at visibility `{$item_vis_descr}`")]
    pub item_span: Span,
    pub item_kind: &'a str,
    pub item_descr: DiagArgFromDisplay<'a>,
    pub item_vis_descr: &'a str,
    #[note("but {$ty_kind} `{$ty_descr}` is only usable at visibility `{$ty_vis_descr}`")]
    pub ty_span: Span,
    pub ty_kind: &'a str,
    pub ty_descr: DiagArgFromDisplay<'a>,
    pub ty_vis_descr: &'a str,
}
