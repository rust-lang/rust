use rustc_errors::codes::*;
use rustc_errors::{DiagArgFromDisplay, MultiSpan};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

#[derive(Diagnostic)]
#[diag(privacy_field_is_private, code = E0451)]
pub(crate) struct FieldIsPrivate {
    #[primary_span]
    pub span: MultiSpan,
    #[label]
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
    #[label(privacy_field_is_private_is_update_syntax_label)]
    IsUpdateSyntax {
        #[primary_span]
        span: Span,
        rest_field_names: String,
        rest_len: usize,
    },
    #[label(privacy_field_is_private_label)]
    Other {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(privacy_item_is_private)]
pub(crate) struct ItemIsPrivate<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
}

#[derive(Diagnostic)]
#[diag(privacy_unnamed_item_is_private)]
pub(crate) struct UnnamedItemIsPrivate {
    #[primary_span]
    pub span: Span,
    pub kind: &'static str,
}

#[derive(Diagnostic)]
#[diag(privacy_in_public_interface, code = E0446)]
pub(crate) struct InPublicInterface<'a> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub vis_descr: &'static str,
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
    #[label(privacy_visibility_label)]
    pub vis_span: Span,
}

#[derive(Diagnostic)]
#[diag(privacy_report_effective_visibility)]
pub(crate) struct ReportEffectiveVisibility {
    #[primary_span]
    pub span: Span,
    pub descr: String,
}

#[derive(LintDiagnostic)]
#[diag(privacy_from_private_dep_in_public_interface)]
pub(crate) struct FromPrivateDependencyInPublicInterface<'a> {
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
    pub krate: Symbol,
}

#[derive(LintDiagnostic)]
#[diag(privacy_unnameable_types_lint)]
pub(crate) struct UnnameableTypesLint<'a> {
    #[label]
    pub span: Span,
    pub kind: &'a str,
    pub descr: DiagArgFromDisplay<'a>,
    pub reachable_vis: &'a str,
    pub reexported_vis: &'a str,
}

// Used for `private_interfaces` and `private_bounds` lints.
// They will replace private-in-public errors and compatibility lints in future.
// See https://rust-lang.github.io/rfcs/2145-type-privacy.html for more details.
#[derive(LintDiagnostic)]
#[diag(privacy_private_interface_or_bounds_lint)]
pub(crate) struct PrivateInterfacesOrBoundsLint<'a> {
    #[label(privacy_item_label)]
    pub item_span: Span,
    pub item_kind: &'a str,
    pub item_descr: DiagArgFromDisplay<'a>,
    pub item_vis_descr: &'a str,
    #[note(privacy_ty_note)]
    pub ty_span: Span,
    pub ty_kind: &'a str,
    pub ty_descr: DiagArgFromDisplay<'a>,
    pub ty_vis_descr: &'a str,
}
