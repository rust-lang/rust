use crate::diagnostics::error::{span_err, throw_span_err, DiagnosticDeriveError};
use proc_macro::Span;
use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens};
use std::collections::{BTreeSet, HashMap};
use std::str::FromStr;
use syn::{spanned::Spanned, Attribute, Meta, Type, TypeTuple};
use synstructure::{BindingInfo, Structure};

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
    fn set_once(&mut self, _: (T, Span));

    fn value(self) -> Option<T>;
}

impl<T> SetOnce<T> for Option<(T, Span)> {
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

    fn value(self) -> Option<T> {
        self.map(|(v, _)| v)
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
pub(crate) fn build_field_mapping<'a>(structure: &Structure<'a>) -> HashMap<String, TokenStream> {
    let mut fields_map = HashMap::new();

    let ast = structure.ast();
    if let syn::Data::Struct(syn::DataStruct { fields, .. }) = &ast.data {
        for field in fields.iter() {
            if let Some(ident) = &field.ident {
                fields_map.insert(ident.to_string(), quote! { &self.#ident });
            }
        }
    }

    fields_map
}
