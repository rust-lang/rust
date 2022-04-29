use crate::diagnostics::error::{span_err, throw_span_err, SessionDiagnosticDeriveError};
use proc_macro::Span;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::BTreeSet;
use std::str::FromStr;
use syn::{spanned::Spanned, Attribute, Meta, Type, Visibility};
use synstructure::BindingInfo;

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

/// Reports an error if the field's type is not `Applicability`.
fn report_error_if_not_applied_to_ty(
    attr: &Attribute,
    info: &FieldInfo<'_>,
    path: &[&str],
    ty_name: &str,
) -> Result<(), SessionDiagnosticDeriveError> {
    if !type_matches_path(&info.ty, path) {
        let name = attr.path.segments.last().unwrap().ident.to_string();
        let name = name.as_str();
        let meta = attr.parse_meta()?;

        throw_span_err!(
            attr.span().unwrap(),
            &format!(
                "the `#[{}{}]` attribute can only be applied to fields of type `{}`",
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

    Ok(())
}

/// Reports an error if the field's type is not `Applicability`.
pub(crate) fn report_error_if_not_applied_to_applicability(
    attr: &Attribute,
    info: &FieldInfo<'_>,
) -> Result<(), SessionDiagnosticDeriveError> {
    report_error_if_not_applied_to_ty(
        attr,
        info,
        &["rustc_errors", "Applicability"],
        "Applicability",
    )
}

/// Reports an error if the field's type is not `Span`.
pub(crate) fn report_error_if_not_applied_to_span(
    attr: &Attribute,
    info: &FieldInfo<'_>,
) -> Result<(), SessionDiagnosticDeriveError> {
    report_error_if_not_applied_to_ty(attr, info, &["rustc_span", "Span"], "Span")
}

/// If `ty` is an Option, returns `Some(inner type)`, otherwise returns `None`.
pub(crate) fn option_inner_ty(ty: &Type) -> Option<&Type> {
    if type_matches_path(ty, &["std", "option", "Option"]) {
        if let Type::Path(ty_path) = ty {
            let path = &ty_path.path;
            let ty = path.segments.iter().last().unwrap();
            if let syn::PathArguments::AngleBracketed(bracketed) = &ty.arguments {
                if bracketed.args.len() == 1 {
                    if let syn::GenericArgument::Type(ty) = &bracketed.args[0] {
                        return Some(ty);
                    }
                }
            }
        }
    }
    None
}

/// Field information passed to the builder. Deliberately omits attrs to discourage the
/// `generate_*` methods from walking the attributes themselves.
pub(crate) struct FieldInfo<'a> {
    pub(crate) vis: &'a Visibility,
    pub(crate) binding: &'a BindingInfo<'a>,
    pub(crate) ty: &'a Type,
    pub(crate) span: &'a proc_macro2::Span,
}

/// Small helper trait for abstracting over `Option` fields that contain a value and a `Span`
/// for error reporting if they are set more than once.
pub(crate) trait SetOnce<T> {
    fn set_once(&mut self, value: T);
}

impl<T> SetOnce<(T, Span)> for Option<(T, Span)> {
    fn set_once(&mut self, (value, span): (T, Span)) {
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
}

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
            if c == '{' && *it.peek().unwrap_or(&'\0') != '{' {
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
