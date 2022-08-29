#![deny(unused_must_use)]

use crate::diagnostics::error::{
    span_err, throw_invalid_attr, throw_invalid_nested_attr, throw_span_err, DiagnosticDeriveError,
};
use crate::diagnostics::utils::{
    report_error_if_not_applied_to_applicability, report_error_if_not_applied_to_span,
    Applicability, FieldInfo, FieldInnerTy, HasFieldMap, SetOnce,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use syn::{parse_quote, spanned::Spanned, Meta, MetaList, MetaNameValue, NestedMeta, Path};
use synstructure::{BindingInfo, Structure, VariantInfo};

/// Which kind of suggestion is being created?
#[derive(Clone, Copy)]
enum SubdiagnosticSuggestionKind {
    /// `#[suggestion]`
    Normal,
    /// `#[suggestion_short]`
    Short,
    /// `#[suggestion_hidden]`
    Hidden,
    /// `#[suggestion_verbose]`
    Verbose,
}

/// Which kind of subdiagnostic is being created from a variant?
#[derive(Clone, Copy)]
enum SubdiagnosticKind {
    /// `#[label(...)]`
    Label,
    /// `#[note(...)]`
    Note,
    /// `#[help(...)]`
    Help,
    /// `#[warning(...)]`
    Warn,
    /// `#[suggestion{,_short,_hidden,_verbose}]`
    Suggestion(SubdiagnosticSuggestionKind),
}

impl FromStr for SubdiagnosticKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "label" => Ok(SubdiagnosticKind::Label),
            "note" => Ok(SubdiagnosticKind::Note),
            "help" => Ok(SubdiagnosticKind::Help),
            "warning" => Ok(SubdiagnosticKind::Warn),
            "suggestion" => Ok(SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Normal)),
            "suggestion_short" => {
                Ok(SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Short))
            }
            "suggestion_hidden" => {
                Ok(SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Hidden))
            }
            "suggestion_verbose" => {
                Ok(SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Verbose))
            }
            _ => Err(()),
        }
    }
}

impl quote::IdentFragment for SubdiagnosticKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SubdiagnosticKind::Label => write!(f, "label"),
            SubdiagnosticKind::Note => write!(f, "note"),
            SubdiagnosticKind::Help => write!(f, "help"),
            SubdiagnosticKind::Warn => write!(f, "warn"),
            SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Normal) => {
                write!(f, "suggestion")
            }
            SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Short) => {
                write!(f, "suggestion_short")
            }
            SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Hidden) => {
                write!(f, "suggestion_hidden")
            }
            SubdiagnosticKind::Suggestion(SubdiagnosticSuggestionKind::Verbose) => {
                write!(f, "suggestion_verbose")
            }
        }
    }

    fn span(&self) -> Option<proc_macro2::Span> {
        None
    }
}

/// The central struct for constructing the `add_to_diagnostic` method from an annotated struct.
pub(crate) struct SessionSubdiagnosticDerive<'a> {
    structure: Structure<'a>,
    diag: syn::Ident,
}

impl<'a> SessionSubdiagnosticDerive<'a> {
    pub(crate) fn new(structure: Structure<'a>) -> Self {
        let diag = format_ident!("diag");
        Self { structure, diag }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let SessionSubdiagnosticDerive { mut structure, diag } = self;
        let implementation = {
            let ast = structure.ast();
            let span = ast.span().unwrap();
            match ast.data {
                syn::Data::Struct(..) | syn::Data::Enum(..) => (),
                syn::Data::Union(..) => {
                    span_err(
                        span,
                        "`#[derive(SessionSubdiagnostic)]` can only be used on structs and enums",
                    );
                }
            }

            if matches!(ast.data, syn::Data::Enum(..)) {
                for attr in &ast.attrs {
                    span_err(
                        attr.span().unwrap(),
                        "unsupported type attribute for subdiagnostic enum",
                    )
                    .emit();
                }
            }

            structure.bind_with(|_| synstructure::BindStyle::Move);
            let variants_ = structure.each_variant(|variant| {
                // Build the mapping of field names to fields. This allows attributes to peek
                // values from other fields.
                let mut fields_map = HashMap::new();
                for binding in variant.bindings() {
                    let field = binding.ast();
                    if let Some(ident) = &field.ident {
                        fields_map.insert(ident.to_string(), quote! { #binding });
                    }
                }

                let mut builder = SessionSubdiagnosticDeriveBuilder {
                    diag: &diag,
                    variant,
                    span,
                    fields: fields_map,
                    kind: None,
                    slug: None,
                    code: None,
                    span_field: None,
                    applicability: None,
                };
                builder.into_tokens().unwrap_or_else(|v| v.to_compile_error())
            });

            quote! {
                match self {
                    #variants_
                }
            }
        };

        let ret = structure.gen_impl(quote! {
            gen impl rustc_errors::AddSubdiagnostic for @Self {
                fn add_to_diagnostic(self, #diag: &mut rustc_errors::Diagnostic) {
                    use rustc_errors::{Applicability, IntoDiagnosticArg};
                    #implementation
                }
            }
        });
        ret
    }
}

/// Tracks persistent information required for building up the call to add to the diagnostic
/// for the final generated method. This is a separate struct to `SessionSubdiagnosticDerive`
/// only to be able to destructure and split `self.builder` and the `self.structure` up to avoid a
/// double mut borrow later on.
struct SessionSubdiagnosticDeriveBuilder<'a> {
    /// The identifier to use for the generated `DiagnosticBuilder` instance.
    diag: &'a syn::Ident,

    /// Info for the current variant (or the type if not an enum).
    variant: &'a VariantInfo<'a>,
    /// Span for the entire type.
    span: proc_macro::Span,

    /// Store a map of field name to its corresponding field. This is built on construction of the
    /// derive builder.
    fields: HashMap<String, TokenStream>,

    /// Subdiagnostic kind of the type/variant.
    kind: Option<(SubdiagnosticKind, proc_macro::Span)>,

    /// Slug of the subdiagnostic - corresponds to the Fluent identifier for the message - from the
    /// `#[kind(slug)]` attribute on the type or variant.
    slug: Option<(Path, proc_macro::Span)>,
    /// If a suggestion, the code to suggest as a replacement - from the `#[kind(code = "...")]`
    /// attribute on the type or variant.
    code: Option<(TokenStream, proc_macro::Span)>,

    /// Identifier for the binding to the `#[primary_span]` field.
    span_field: Option<(proc_macro2::Ident, proc_macro::Span)>,
    /// If a suggestion, the identifier for the binding to the `#[applicability]` field or a
    /// `rustc_errors::Applicability::*` variant directly.
    applicability: Option<(TokenStream, proc_macro::Span)>,
}

impl<'a> HasFieldMap for SessionSubdiagnosticDeriveBuilder<'a> {
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream> {
        self.fields.get(field)
    }
}

impl<'a> SessionSubdiagnosticDeriveBuilder<'a> {
    fn identify_kind(&mut self) -> Result<(), DiagnosticDeriveError> {
        for attr in self.variant.ast().attrs {
            let span = attr.span().unwrap();

            let name = attr.path.segments.last().unwrap().ident.to_string();
            let name = name.as_str();

            let meta = attr.parse_meta()?;
            let kind = match meta {
                Meta::List(MetaList { ref nested, .. }) => {
                    let mut nested_iter = nested.into_iter();
                    if let Some(nested_attr) = nested_iter.next() {
                        match nested_attr {
                            NestedMeta::Meta(Meta::Path(path)) => {
                                self.slug.set_once((path.clone(), span));
                            }
                            NestedMeta::Meta(meta @ Meta::NameValue(_))
                                if matches!(
                                    meta.path().segments.last().unwrap().ident.to_string().as_str(),
                                    "code" | "applicability"
                                ) =>
                            {
                                // don't error for valid follow-up attributes
                            }
                            nested_attr => {
                                throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                                    diag.help(
                                        "first argument of the attribute should be the diagnostic \
                                         slug",
                                    )
                                })
                            }
                        };
                    }

                    for nested_attr in nested_iter {
                        let meta = match nested_attr {
                            NestedMeta::Meta(ref meta) => meta,
                            _ => throw_invalid_nested_attr!(attr, &nested_attr),
                        };

                        let span = meta.span().unwrap();
                        let nested_name = meta.path().segments.last().unwrap().ident.to_string();
                        let nested_name = nested_name.as_str();

                        match meta {
                            Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(s), .. }) => {
                                match nested_name {
                                    "code" => {
                                        let formatted_str = self.build_format(&s.value(), s.span());
                                        self.code.set_once((formatted_str, span));
                                    }
                                    "applicability" => {
                                        let value = match Applicability::from_str(&s.value()) {
                                            Ok(v) => v,
                                            Err(()) => {
                                                span_err(span, "invalid applicability").emit();
                                                Applicability::Unspecified
                                            }
                                        };
                                        self.applicability.set_once((quote! { #value }, span));
                                    }
                                    _ => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                                        diag.help(
                                            "only `code` and `applicability` are valid nested \
                                             attributes",
                                        )
                                    }),
                                }
                            }
                            _ => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                                if matches!(meta, Meta::Path(_)) {
                                    diag.help(
                                        "a diagnostic slug must be the first argument to the \
                                         attribute",
                                    )
                                } else {
                                    diag
                                }
                            }),
                        }
                    }

                    let Ok(kind) = SubdiagnosticKind::from_str(name) else {
                        throw_invalid_attr!(attr, &meta)
                    };

                    kind
                }
                _ => throw_invalid_attr!(attr, &meta),
            };

            if matches!(
                kind,
                SubdiagnosticKind::Label | SubdiagnosticKind::Help | SubdiagnosticKind::Note
            ) && self.code.is_some()
            {
                throw_span_err!(
                    span,
                    &format!("`code` is not a valid nested attribute of a `{}` attribute", name)
                );
            }

            if matches!(
                kind,
                SubdiagnosticKind::Label | SubdiagnosticKind::Help | SubdiagnosticKind::Note
            ) && self.applicability.is_some()
            {
                throw_span_err!(
                    span,
                    &format!(
                        "`applicability` is not a valid nested attribute of a `{}` attribute",
                        name
                    )
                );
            }

            if self.slug.is_none() {
                throw_span_err!(
                    span,
                    &format!(
                        "diagnostic slug must be first argument of a `#[{}(...)]` attribute",
                        name
                    )
                );
            }

            self.kind.set_once((kind, span));
        }

        Ok(())
    }

    fn generate_field_code(
        &mut self,
        binding: &BindingInfo<'_>,
        is_suggestion: bool,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        let ast = binding.ast();

        let inner_ty = FieldInnerTy::from_type(&ast.ty);
        let info = FieldInfo {
            binding: binding,
            ty: inner_ty.inner_type().unwrap_or(&ast.ty),
            span: &ast.span(),
        };

        for attr in &ast.attrs {
            let name = attr.path.segments.last().unwrap().ident.to_string();
            let name = name.as_str();
            let span = attr.span().unwrap();

            let meta = attr.parse_meta()?;
            match meta {
                Meta::Path(_) => match name {
                    "primary_span" => {
                        report_error_if_not_applied_to_span(attr, &info)?;
                        self.span_field.set_once((binding.binding.clone(), span));
                        return Ok(quote! {});
                    }
                    "applicability" if is_suggestion => {
                        report_error_if_not_applied_to_applicability(attr, &info)?;
                        let binding = binding.binding.clone();
                        self.applicability.set_once((quote! { #binding }, span));
                        return Ok(quote! {});
                    }
                    "applicability" => {
                        span_err(span, "`#[applicability]` is only valid on suggestions").emit();
                        return Ok(quote! {});
                    }
                    "skip_arg" => {
                        return Ok(quote! {});
                    }
                    _ => throw_invalid_attr!(attr, &meta, |diag| {
                        diag.help(
                            "only `primary_span`, `applicability` and `skip_arg` are valid field \
                             attributes",
                        )
                    }),
                },
                _ => throw_invalid_attr!(attr, &meta),
            }
        }

        let ident = ast.ident.as_ref().unwrap();

        let diag = &self.diag;
        let generated = quote! {
            #diag.set_arg(
                stringify!(#ident),
                #binding
            );
        };

        Ok(inner_ty.with(binding, generated))
    }

    fn into_tokens(&mut self) -> Result<TokenStream, DiagnosticDeriveError> {
        self.identify_kind()?;
        let Some(kind) = self.kind.map(|(kind, _)| kind) else {
            throw_span_err!(
                self.variant.ast().ident.span().unwrap(),
                "subdiagnostic kind not specified"
            );
        };

        let is_suggestion = matches!(kind, SubdiagnosticKind::Suggestion(_));

        let mut args = TokenStream::new();
        for binding in self.variant.bindings() {
            let arg = self
                .generate_field_code(binding, is_suggestion)
                .unwrap_or_else(|v| v.to_compile_error());
            args.extend(arg);
        }

        // Missing slug errors will already have been reported.
        let slug = self
            .slug
            .as_ref()
            .map(|(slug, _)| slug.clone())
            .unwrap_or_else(|| parse_quote! { you::need::to::specify::a::slug });
        let code = match self.code.as_ref() {
            Some((code, _)) => Some(quote! { #code }),
            None if is_suggestion => {
                span_err(self.span, "suggestion without `code = \"...\"`").emit();
                Some(quote! { /* macro error */ "..." })
            }
            None => None,
        };

        let span_field = self.span_field.as_ref().map(|(span, _)| span);
        let applicability = match self.applicability.clone() {
            Some((applicability, _)) => Some(applicability),
            None if is_suggestion => {
                span_err(self.span, "suggestion without `applicability`").emit();
                Some(quote! { rustc_errors::Applicability::Unspecified })
            }
            None => None,
        };

        let diag = &self.diag;
        let name = format_ident!("{}{}", if span_field.is_some() { "span_" } else { "" }, kind);
        let message = quote! { rustc_errors::fluent::#slug };
        let call = if matches!(kind, SubdiagnosticKind::Suggestion(..)) {
            if let Some(span) = span_field {
                quote! { #diag.#name(#span, #message, #code, #applicability); }
            } else {
                span_err(self.span, "suggestion without `#[primary_span]` field").emit();
                quote! { unreachable!(); }
            }
        } else if matches!(kind, SubdiagnosticKind::Label) {
            if let Some(span) = span_field {
                quote! { #diag.#name(#span, #message); }
            } else {
                span_err(self.span, "label without `#[primary_span]` field").emit();
                quote! { unreachable!(); }
            }
        } else {
            if let Some(span) = span_field {
                quote! { #diag.#name(#span, #message); }
            } else {
                quote! { #diag.#name(#message); }
            }
        };

        Ok(quote! {
            #call
            #args
        })
    }
}
