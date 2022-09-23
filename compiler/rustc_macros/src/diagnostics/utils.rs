use crate::diagnostics::error::{
    span_err, throw_invalid_attr, throw_invalid_nested_attr, throw_span_err, DiagnosticDeriveError,
};
use proc_macro::Span;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::str::FromStr;
use syn::{spanned::Spanned, Attribute, Field, Meta, Type, TypeTuple};
use syn::{MetaList, MetaNameValue, NestedMeta, Path};
use synstructure::{BindStyle, BindingInfo, VariantInfo};

use super::error::invalid_nested_attr;

/// Checks whether the type name of `ty` matches `name`.
///
/// Given some struct at `a::b::c::Foo`, this will return true for `c::Foo`, `b::c::Foo`, or
/// `a::b::c::Foo`. This reasonably allows qualified names to be used in the macro.
pub(crate) fn type_matches_path(ty: &Type, name: &[&str]) -> bool {
    if let Type::Path(ty) = ty {
        ty.path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .rev()
            .zip(name.iter().rev())
            .all(|(x, y)| &x.as_str() == y)
    } else {
        false
    }
}

/// Checks whether the type `ty` is `()`.
pub(crate) fn type_is_unit(ty: &Type) -> bool {
    if let Type::Tuple(TypeTuple { elems, .. }) = ty { elems.is_empty() } else { false }
}

/// Reports a type error for field with `attr`.
pub(crate) fn report_type_error(
    attr: &Attribute,
    ty_name: &str,
) -> Result<!, DiagnosticDeriveError> {
    let name = attr.path.segments.last().unwrap().ident.to_string();
    let meta = attr.parse_meta()?;

    throw_span_err!(
        attr.span().unwrap(),
        &format!(
            "the `#[{}{}]` attribute can only be applied to fields of type {}",
            name,
            match meta {
                Meta::Path(_) => "",
                Meta::NameValue(_) => " = ...",
                Meta::List(_) => "(...)",
            },
            ty_name
        )
    );
}

/// Reports an error if the field's type does not match `path`.
fn report_error_if_not_applied_to_ty(
    attr: &Attribute,
    info: &FieldInfo<'_>,
    path: &[&str],
    ty_name: &str,
) -> Result<(), DiagnosticDeriveError> {
    if !type_matches_path(&info.ty, path) {
        report_type_error(attr, ty_name)?;
    }

    Ok(())
}

/// Reports an error if the field's type is not `Applicability`.
pub(crate) fn report_error_if_not_applied_to_applicability(
    attr: &Attribute,
    info: &FieldInfo<'_>,
) -> Result<(), DiagnosticDeriveError> {
    report_error_if_not_applied_to_ty(
        attr,
        info,
        &["rustc_errors", "Applicability"],
        "`Applicability`",
    )
}

/// Reports an error if the field's type is not `Span`.
pub(crate) fn report_error_if_not_applied_to_span(
    attr: &Attribute,
    info: &FieldInfo<'_>,
) -> Result<(), DiagnosticDeriveError> {
    if !type_matches_path(&info.ty, &["rustc_span", "Span"])
        && !type_matches_path(&info.ty, &["rustc_errors", "MultiSpan"])
    {
        report_type_error(attr, "`Span` or `MultiSpan`")?;
    }

    Ok(())
}

/// Inner type of a field and type of wrapper.
pub(crate) enum FieldInnerTy<'ty> {
    /// Field is wrapped in a `Option<$inner>`.
    Option(&'ty Type),
    /// Field is wrapped in a `Vec<$inner>`.
    Vec(&'ty Type),
    /// Field isn't wrapped in an outer type.
    None,
}

impl<'ty> FieldInnerTy<'ty> {
    /// Returns inner type for a field, if there is one.
    ///
    /// - If `ty` is an `Option`, returns `FieldInnerTy::Option { inner: (inner type) }`.
    /// - If `ty` is a `Vec`, returns `FieldInnerTy::Vec { inner: (inner type) }`.
    /// - Otherwise returns `None`.
    pub(crate) fn from_type(ty: &'ty Type) -> Self {
        let variant: &dyn Fn(&'ty Type) -> FieldInnerTy<'ty> =
            if type_matches_path(ty, &["std", "option", "Option"]) {
                &FieldInnerTy::Option
            } else if type_matches_path(ty, &["std", "vec", "Vec"]) {
                &FieldInnerTy::Vec
            } else {
                return FieldInnerTy::None;
            };

        if let Type::Path(ty_path) = ty {
            let path = &ty_path.path;
            let ty = path.segments.iter().last().unwrap();
            if let syn::PathArguments::AngleBracketed(bracketed) = &ty.arguments {
                if bracketed.args.len() == 1 {
                    if let syn::GenericArgument::Type(ty) = &bracketed.args[0] {
                        return variant(ty);
                    }
                }
            }
        }

        unreachable!();
    }

    /// Returns `Option` containing inner type if there is one.
    pub(crate) fn inner_type(&self) -> Option<&'ty Type> {
        match self {
            FieldInnerTy::Option(inner) | FieldInnerTy::Vec(inner) => Some(inner),
            FieldInnerTy::None => None,
        }
    }

    /// Surrounds `inner` with destructured wrapper type, exposing inner type as `binding`.
    pub(crate) fn with(&self, binding: impl ToTokens, inner: impl ToTokens) -> TokenStream {
        match self {
            FieldInnerTy::Option(..) => quote! {
                if let Some(#binding) = #binding {
                    #inner
                }
            },
            FieldInnerTy::Vec(..) => quote! {
                for #binding in #binding {
                    #inner
                }
            },
            FieldInnerTy::None => quote! { #inner },
        }
    }
}

/// Field information passed to the builder. Deliberately omits attrs to discourage the
/// `generate_*` methods from walking the attributes themselves.
pub(crate) struct FieldInfo<'a> {
    pub(crate) binding: &'a BindingInfo<'a>,
    pub(crate) ty: &'a Type,
    pub(crate) span: &'a proc_macro2::Span,
}

/// Small helper trait for abstracting over `Option` fields that contain a value and a `Span`
/// for error reporting if they are set more than once.
pub(crate) trait SetOnce<T> {
    fn set_once(&mut self, value: T, span: Span);

    fn value(self) -> Option<T>;
    fn value_ref(&self) -> Option<&T>;
}

/// An [`Option<T>`] that keeps track of the span that caused it to be set; used with [`SetOnce`].
pub(super) type SpannedOption<T> = Option<(T, Span)>;

impl<T> SetOnce<T> for SpannedOption<T> {
    fn set_once(&mut self, value: T, span: Span) {
        match self {
            None => {
                *self = Some((value, span));
            }
            Some((_, prev_span)) => {
                span_err(span, "specified multiple times")
                    .span_note(*prev_span, "previously specified here")
                    .emit();
            }
        }
    }

    fn value(self) -> Option<T> {
        self.map(|(v, _)| v)
    }

    fn value_ref(&self) -> Option<&T> {
        self.as_ref().map(|(v, _)| v)
    }
}

pub(super) type FieldMap = HashMap<String, TokenStream>;

pub(crate) trait HasFieldMap {
    /// Returns the binding for the field with the given name, if it exists on the type.
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream>;

    /// In the strings in the attributes supplied to this macro, we want callers to be able to
    /// reference fields in the format string. For example:
    ///
    /// ```ignore (not-usage-example)
    /// /// Suggest `==` when users wrote `===`.
    /// #[suggestion(slug = "parser-not-javascript-eq", code = "{lhs} == {rhs}")]
    /// struct NotJavaScriptEq {
    ///     #[primary_span]
    ///     span: Span,
    ///     lhs: Ident,
    ///     rhs: Ident,
    /// }
    /// ```
    ///
    /// We want to automatically pick up that `{lhs}` refers `self.lhs` and `{rhs}` refers to
    /// `self.rhs`, then generate this call to `format!`:
    ///
    /// ```ignore (not-usage-example)
    /// format!("{lhs} == {rhs}", lhs = self.lhs, rhs = self.rhs)
    /// ```
    ///
    /// This function builds the entire call to `format!`.
    fn build_format(&self, input: &str, span: proc_macro2::Span) -> TokenStream {
        // This set is used later to generate the final format string. To keep builds reproducible,
        // the iteration order needs to be deterministic, hence why we use a `BTreeSet` here
        // instead of a `HashSet`.
        let mut referenced_fields: BTreeSet<String> = BTreeSet::new();

        // At this point, we can start parsing the format string.
        let mut it = input.chars().peekable();

        // Once the start of a format string has been found, process the format string and spit out
        // the referenced fields. Leaves `it` sitting on the closing brace of the format string, so
        // the next call to `it.next()` retrieves the next character.
        while let Some(c) = it.next() {
            if c != '{' {
                continue;
            }
            if *it.peek().unwrap_or(&'\0') == '{' {
                assert_eq!(it.next().unwrap(), '{');
                continue;
            }
            let mut eat_argument = || -> Option<String> {
                let mut result = String::new();
                // Format specifiers look like:
                //
                //   format   := '{' [ argument ] [ ':' format_spec ] '}' .
                //
                // Therefore, we only need to eat until ':' or '}' to find the argument.
                while let Some(c) = it.next() {
                    result.push(c);
                    let next = *it.peek().unwrap_or(&'\0');
                    if next == '}' {
                        break;
                    } else if next == ':' {
                        // Eat the ':' character.
                        assert_eq!(it.next().unwrap(), ':');
                        break;
                    }
                }
                // Eat until (and including) the matching '}'
                while it.next()? != '}' {
                    continue;
                }
                Some(result)
            };

            if let Some(referenced_field) = eat_argument() {
                referenced_fields.insert(referenced_field);
            }
        }

        // At this point, `referenced_fields` contains a set of the unique fields that were
        // referenced in the format string. Generate the corresponding "x = self.x" format
        // string parameters:
        let args = referenced_fields.into_iter().map(|field: String| {
            let field_ident = format_ident!("{}", field);
            let value = match self.get_field_binding(&field) {
                Some(value) => value.clone(),
                // This field doesn't exist. Emit a diagnostic.
                None => {
                    span_err(
                        span.unwrap(),
                        &format!("`{}` doesn't refer to a field on this type", field),
                    )
                    .emit();
                    quote! {
                        "{#field}"
                    }
                }
            };
            quote! {
                #field_ident = #value
            }
        });
        quote! {
            format!(#input #(,#args)*)
        }
    }
}

/// `Applicability` of a suggestion - mirrors `rustc_errors::Applicability` - and used to represent
/// the user's selection of applicability if specified in an attribute.
#[derive(Clone, Copy)]
pub(crate) enum Applicability {
    MachineApplicable,
    MaybeIncorrect,
    HasPlaceholders,
    Unspecified,
}

impl FromStr for Applicability {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "machine-applicable" => Ok(Applicability::MachineApplicable),
            "maybe-incorrect" => Ok(Applicability::MaybeIncorrect),
            "has-placeholders" => Ok(Applicability::HasPlaceholders),
            "unspecified" => Ok(Applicability::Unspecified),
            _ => Err(()),
        }
    }
}

impl quote::ToTokens for Applicability {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(match self {
            Applicability::MachineApplicable => {
                quote! { rustc_errors::Applicability::MachineApplicable }
            }
            Applicability::MaybeIncorrect => {
                quote! { rustc_errors::Applicability::MaybeIncorrect }
            }
            Applicability::HasPlaceholders => {
                quote! { rustc_errors::Applicability::HasPlaceholders }
            }
            Applicability::Unspecified => {
                quote! { rustc_errors::Applicability::Unspecified }
            }
        });
    }
}

/// Build the mapping of field names to fields. This allows attributes to peek values from
/// other fields.
pub(super) fn build_field_mapping<'v>(variant: &VariantInfo<'v>) -> HashMap<String, TokenStream> {
    let mut fields_map = FieldMap::new();
    for binding in variant.bindings() {
        if let Some(ident) = &binding.ast().ident {
            fields_map.insert(ident.to_string(), quote! { #binding });
        }
    }
    fields_map
}

/// Possible styles for suggestion subdiagnostics.
#[derive(Clone, Copy)]
pub(super) enum SuggestionKind {
    /// `#[suggestion]`
    Normal,
    /// `#[suggestion_short]`
    Short,
    /// `#[suggestion_hidden]`
    Hidden,
    /// `#[suggestion_verbose]`
    Verbose,
}

impl FromStr for SuggestionKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "" => Ok(SuggestionKind::Normal),
            "_short" => Ok(SuggestionKind::Short),
            "_hidden" => Ok(SuggestionKind::Hidden),
            "_verbose" => Ok(SuggestionKind::Verbose),
            _ => Err(()),
        }
    }
}

impl SuggestionKind {
    pub fn to_suggestion_style(&self) -> TokenStream {
        match self {
            SuggestionKind::Normal => {
                quote! { rustc_errors::SuggestionStyle::ShowCode }
            }
            SuggestionKind::Short => {
                quote! { rustc_errors::SuggestionStyle::HideCodeInline }
            }
            SuggestionKind::Hidden => {
                quote! { rustc_errors::SuggestionStyle::HideCodeAlways }
            }
            SuggestionKind::Verbose => {
                quote! { rustc_errors::SuggestionStyle::ShowAlways }
            }
        }
    }
}

/// Types of subdiagnostics that can be created using attributes
#[derive(Clone)]
pub(super) enum SubdiagnosticKind {
    /// `#[label(...)]`
    Label,
    /// `#[note(...)]`
    Note,
    /// `#[help(...)]`
    Help,
    /// `#[warning(...)]`
    Warn,
    /// `#[suggestion{,_short,_hidden,_verbose}]`
    Suggestion {
        suggestion_kind: SuggestionKind,
        applicability: SpannedOption<Applicability>,
        code: TokenStream,
    },
    /// `#[multipart_suggestion{,_short,_hidden,_verbose}]`
    MultipartSuggestion {
        suggestion_kind: SuggestionKind,
        applicability: SpannedOption<Applicability>,
    },
}

impl SubdiagnosticKind {
    /// Constructs a `SubdiagnosticKind` from a field or type attribute such as `#[note]`,
    /// `#[error(parser::add_paren)]` or `#[suggestion(code = "...")]`. Returns the
    /// `SubdiagnosticKind` and the diagnostic slug, if specified.
    pub(super) fn from_attr(
        attr: &Attribute,
        fields: &impl HasFieldMap,
    ) -> Result<(SubdiagnosticKind, Option<Path>), DiagnosticDeriveError> {
        let span = attr.span().unwrap();

        let name = attr.path.segments.last().unwrap().ident.to_string();
        let name = name.as_str();

        let meta = attr.parse_meta()?;
        let mut kind = match name {
            "label" => SubdiagnosticKind::Label,
            "note" => SubdiagnosticKind::Note,
            "help" => SubdiagnosticKind::Help,
            "warning" => SubdiagnosticKind::Warn,
            _ => {
                if let Some(suggestion_kind) =
                    name.strip_prefix("suggestion").and_then(|s| s.parse().ok())
                {
                    SubdiagnosticKind::Suggestion {
                        suggestion_kind,
                        applicability: None,
                        code: TokenStream::new(),
                    }
                } else if let Some(suggestion_kind) =
                    name.strip_prefix("multipart_suggestion").and_then(|s| s.parse().ok())
                {
                    SubdiagnosticKind::MultipartSuggestion { suggestion_kind, applicability: None }
                } else {
                    throw_invalid_attr!(attr, &meta);
                }
            }
        };

        let nested = match meta {
            Meta::List(MetaList { ref nested, .. }) => {
                // An attribute with properties, such as `#[suggestion(code = "...")]` or
                // `#[error(some::slug)]`
                nested
            }
            Meta::Path(_) => {
                // An attribute without a slug or other properties, such as `#[note]` - return
                // without further processing.
                //
                // Only allow this if there are no mandatory properties, such as `code = "..."` in
                // `#[suggestion(...)]`
                match kind {
                    SubdiagnosticKind::Label
                    | SubdiagnosticKind::Note
                    | SubdiagnosticKind::Help
                    | SubdiagnosticKind::Warn
                    | SubdiagnosticKind::MultipartSuggestion { .. } => return Ok((kind, None)),
                    SubdiagnosticKind::Suggestion { .. } => {
                        throw_span_err!(span, "suggestion without `code = \"...\"`")
                    }
                }
            }
            _ => {
                throw_invalid_attr!(attr, &meta)
            }
        };

        let mut code = None;

        let mut nested_iter = nested.into_iter().peekable();

        // Peek at the first nested attribute: if it's a slug path, consume it.
        let slug = if let Some(NestedMeta::Meta(Meta::Path(path))) = nested_iter.peek() {
            let path = path.clone();
            // Advance the iterator.
            nested_iter.next();
            Some(path)
        } else {
            None
        };

        for nested_attr in nested_iter {
            let meta = match nested_attr {
                NestedMeta::Meta(ref meta) => meta,
                NestedMeta::Lit(_) => {
                    invalid_nested_attr(attr, &nested_attr).emit();
                    continue;
                }
            };

            let span = meta.span().unwrap();
            let nested_name = meta.path().segments.last().unwrap().ident.to_string();
            let nested_name = nested_name.as_str();

            let value = match meta {
                Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(value), .. }) => value,
                Meta::Path(_) => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                    diag.help("a diagnostic slug must be the first argument to the attribute")
                }),
                _ => {
                    invalid_nested_attr(attr, &nested_attr).emit();
                    continue;
                }
            };

            match (nested_name, &mut kind) {
                ("code", SubdiagnosticKind::Suggestion { .. }) => {
                    let formatted_str = fields.build_format(&value.value(), value.span());
                    code.set_once(formatted_str, span);
                }
                (
                    "applicability",
                    SubdiagnosticKind::Suggestion { ref mut applicability, .. }
                    | SubdiagnosticKind::MultipartSuggestion { ref mut applicability, .. },
                ) => {
                    let value = Applicability::from_str(&value.value()).unwrap_or_else(|()| {
                        span_err(span, "invalid applicability").emit();
                        Applicability::Unspecified
                    });
                    applicability.set_once(value, span);
                }

                // Invalid nested attribute
                (_, SubdiagnosticKind::Suggestion { .. }) => {
                    invalid_nested_attr(attr, &nested_attr)
                        .help("only `code` and `applicability` are valid nested attributes")
                        .emit();
                }
                (_, SubdiagnosticKind::MultipartSuggestion { .. }) => {
                    invalid_nested_attr(attr, &nested_attr)
                        .help("only `applicability` is a valid nested attributes")
                        .emit()
                }
                _ => {
                    invalid_nested_attr(attr, &nested_attr).emit();
                }
            }
        }

        match kind {
            SubdiagnosticKind::Suggestion { code: ref mut code_field, .. } => {
                *code_field = if let Some((code, _)) = code {
                    code
                } else {
                    span_err(span, "suggestion without `code = \"...\"`").emit();
                    quote! { "" }
                }
            }
            SubdiagnosticKind::Label
            | SubdiagnosticKind::Note
            | SubdiagnosticKind::Help
            | SubdiagnosticKind::Warn
            | SubdiagnosticKind::MultipartSuggestion { .. } => {}
        }

        Ok((kind, slug))
    }
}

impl quote::IdentFragment for SubdiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubdiagnosticKind::Label => write!(f, "label"),
            SubdiagnosticKind::Note => write!(f, "note"),
            SubdiagnosticKind::Help => write!(f, "help"),
            SubdiagnosticKind::Warn => write!(f, "warn"),
            SubdiagnosticKind::Suggestion { .. } => write!(f, "suggestion_with_style"),
            SubdiagnosticKind::MultipartSuggestion { .. } => {
                write!(f, "multipart_suggestion_with_style")
            }
        }
    }

    fn span(&self) -> Option<proc_macro2::Span> {
        None
    }
}

/// Wrapper around `synstructure::BindStyle` which implements `Ord`.
#[derive(PartialEq, Eq)]
pub(super) struct OrderedBindStyle(pub(super) BindStyle);

impl OrderedBindStyle {
    /// Is `BindStyle::Move` or `BindStyle::MoveMut`?
    pub(super) fn is_move(&self) -> bool {
        matches!(self.0, BindStyle::Move | BindStyle::MoveMut)
    }
}

impl Ord for OrderedBindStyle {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_move(), other.is_move()) {
            // If both `self` and `other` are the same, then ordering is equal.
            (true, true) | (false, false) => Ordering::Equal,
            // If `self` is not a move then it should be considered less than `other` (so that
            // references are sorted first).
            (false, _) => Ordering::Less,
            // If `self` is a move then it must be greater than `other` (again, so that references
            // are sorted first).
            (true, _) => Ordering::Greater,
        }
    }
}

impl PartialOrd for OrderedBindStyle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns `true` if `field` should generate a `set_arg` call rather than any other diagnostic
/// call (like `span_label`).
pub(super) fn should_generate_set_arg(field: &Field) -> bool {
    field.attrs.is_empty()
}

/// Returns `true` if `field` needs to have code generated in the by-move branch of the
/// generated derive rather than the by-ref branch.
pub(super) fn bind_style_of_field(field: &Field) -> OrderedBindStyle {
    let generates_set_arg = should_generate_set_arg(field);
    let is_multispan = type_matches_path(&field.ty, &["rustc_errors", "MultiSpan"]);
    // FIXME(davidtwco): better support for one field needing to be in the by-move and
    // by-ref branches.
    let is_subdiagnostic = field
        .attrs
        .iter()
        .map(|attr| attr.path.segments.last().unwrap().ident.to_string())
        .any(|attr| attr == "subdiagnostic");

    // `set_arg` calls take their argument by-move..
    let needs_move = generates_set_arg
        // If this is a `MultiSpan` field then it needs to be moved to be used by any
        // attribute..
        || is_multispan
        // If this a `#[subdiagnostic]` then it needs to be moved as the other diagnostic is
        // unlikely to be `Copy`..
        || is_subdiagnostic;

    OrderedBindStyle(if needs_move { BindStyle::Move } else { BindStyle::Ref })
}
