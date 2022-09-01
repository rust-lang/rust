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
use syn::{spanned::Spanned, Attribute, Meta, MetaList, MetaNameValue, NestedMeta, Path};
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

impl FromStr for SubdiagnosticSuggestionKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "" => Ok(SubdiagnosticSuggestionKind::Normal),
            "_short" => Ok(SubdiagnosticSuggestionKind::Short),
            "_hidden" => Ok(SubdiagnosticSuggestionKind::Hidden),
            "_verbose" => Ok(SubdiagnosticSuggestionKind::Verbose),
            _ => Err(()),
        }
    }
}

impl SubdiagnosticSuggestionKind {
    pub fn to_suggestion_style(&self) -> TokenStream {
        match self {
            SubdiagnosticSuggestionKind::Normal => {
                quote! { rustc_errors::SuggestionStyle::ShowCode }
            }
            SubdiagnosticSuggestionKind::Short => {
                quote! { rustc_errors::SuggestionStyle::HideCodeInline }
            }
            SubdiagnosticSuggestionKind::Hidden => {
                quote! { rustc_errors::SuggestionStyle::HideCodeAlways }
            }
            SubdiagnosticSuggestionKind::Verbose => {
                quote! { rustc_errors::SuggestionStyle::ShowAlways }
            }
        }
    }
}

/// Which kind of subdiagnostic is being created from a variant?
#[derive(Clone)]
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
    Suggestion { suggestion_kind: SubdiagnosticSuggestionKind, code: TokenStream },
    /// `#[multipart_suggestion{,_short,_hidden,_verbose}]`
    MultipartSuggestion { suggestion_kind: SubdiagnosticSuggestionKind },
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
                    span_field: None,
                    applicability: None,
                    has_suggestion_parts: false,
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

    /// Identifier for the binding to the `#[primary_span]` field.
    span_field: Option<(proc_macro2::Ident, proc_macro::Span)>,
    /// If a suggestion, the identifier for the binding to the `#[applicability]` field or a
    /// `rustc_errors::Applicability::*` variant directly.
    applicability: Option<(TokenStream, proc_macro::Span)>,

    /// Set to true when a `#[suggestion_part]` field is encountered, used to generate an error
    /// during finalization if still `false`.
    has_suggestion_parts: bool,
}

impl<'a> HasFieldMap for SessionSubdiagnosticDeriveBuilder<'a> {
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream> {
        self.fields.get(field)
    }
}

/// Provides frequently-needed information about the diagnostic kinds being derived for this type.
#[derive(Clone, Copy, Debug)]
struct KindsStatistics {
    has_multipart_suggestion: bool,
    all_multipart_suggestions: bool,
    has_normal_suggestion: bool,
}

impl<'a> FromIterator<&'a SubdiagnosticKind> for KindsStatistics {
    fn from_iter<T: IntoIterator<Item = &'a SubdiagnosticKind>>(kinds: T) -> Self {
        let mut ret = Self {
            has_multipart_suggestion: false,
            all_multipart_suggestions: true,
            has_normal_suggestion: false,
        };
        for kind in kinds {
            if let SubdiagnosticKind::MultipartSuggestion { .. } = kind {
                ret.has_multipart_suggestion = true;
            } else {
                ret.all_multipart_suggestions = false;
            }

            if let SubdiagnosticKind::Suggestion { .. } = kind {
                ret.has_normal_suggestion = true;
            }
        }
        ret
    }
}

impl<'a> SessionSubdiagnosticDeriveBuilder<'a> {
    fn identify_kind(&mut self) -> Result<Vec<(SubdiagnosticKind, Path)>, DiagnosticDeriveError> {
        let mut kind_slugs = vec![];

        for attr in self.variant.ast().attrs {
            let span = attr.span().unwrap();

            let name = attr.path.segments.last().unwrap().ident.to_string();
            let name = name.as_str();

            let meta = attr.parse_meta()?;
            let Meta::List(MetaList { ref nested, .. }) = meta else {
                throw_invalid_attr!(attr, &meta);
            };

            let mut kind = match name {
                "label" => SubdiagnosticKind::Label,
                "note" => SubdiagnosticKind::Note,
                "help" => SubdiagnosticKind::Help,
                "warning" => SubdiagnosticKind::Warn,
                _ => {
                    if let Some(suggestion_kind) =
                        name.strip_prefix("suggestion").and_then(|s| s.parse().ok())
                    {
                        SubdiagnosticKind::Suggestion { suggestion_kind, code: TokenStream::new() }
                    } else if let Some(suggestion_kind) =
                        name.strip_prefix("multipart_suggestion").and_then(|s| s.parse().ok())
                    {
                        SubdiagnosticKind::MultipartSuggestion { suggestion_kind }
                    } else {
                        throw_invalid_attr!(attr, &meta);
                    }
                }
            };

            let mut slug = None;
            let mut code = None;

            let mut nested_iter = nested.into_iter();
            if let Some(nested_attr) = nested_iter.next() {
                match nested_attr {
                    NestedMeta::Meta(Meta::Path(path)) => {
                        slug.set_once((path.clone(), span));
                    }
                    NestedMeta::Meta(meta @ Meta::NameValue(_))
                        if matches!(
                            meta.path().segments.last().unwrap().ident.to_string().as_str(),
                            "code" | "applicability"
                        ) =>
                    {
                        // Don't error for valid follow-up attributes.
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

                let value = match meta {
                    Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(value), .. }) => value,
                    Meta::Path(_) => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                        diag.help("a diagnostic slug must be the first argument to the attribute")
                    }),
                    _ => throw_invalid_nested_attr!(attr, &nested_attr),
                };

                match nested_name {
                    "code" => {
                        if matches!(kind, SubdiagnosticKind::Suggestion { .. }) {
                            let formatted_str = self.build_format(&value.value(), value.span());
                            code.set_once((formatted_str, span));
                        } else {
                            span_err(
                                span,
                                &format!(
                                    "`code` is not a valid nested attribute of a `{}` attribute",
                                    name
                                ),
                            )
                            .emit();
                        }
                    }
                    "applicability" => {
                        if matches!(
                            kind,
                            SubdiagnosticKind::Suggestion { .. }
                                | SubdiagnosticKind::MultipartSuggestion { .. }
                        ) {
                            let value =
                                Applicability::from_str(&value.value()).unwrap_or_else(|()| {
                                    span_err(span, "invalid applicability").emit();
                                    Applicability::Unspecified
                                });
                            self.applicability.set_once((quote! { #value }, span));
                        } else {
                            span_err(
                                span,
                                &format!(
                                    "`applicability` is not a valid nested attribute of a `{}` attribute",
                                    name
                                )
                            ).emit();
                        }
                    }
                    _ => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                        diag.help("only `code` and `applicability` are valid nested attributes")
                    }),
                }
            }

            let Some((slug, _)) = slug else {
                throw_span_err!(
                    span,
                    &format!(
                        "diagnostic slug must be first argument of a `#[{}(...)]` attribute",
                        name
                    )
                );
            };

            match kind {
                SubdiagnosticKind::Suggestion { code: ref mut code_field, .. } => {
                    let Some((code, _)) = code else {
                        throw_span_err!(span, "suggestion without `code = \"...\"`");
                    };
                    *code_field = code;
                }
                SubdiagnosticKind::Label
                | SubdiagnosticKind::Note
                | SubdiagnosticKind::Help
                | SubdiagnosticKind::Warn
                | SubdiagnosticKind::MultipartSuggestion { .. } => {}
            }

            kind_slugs.push((kind, slug))
        }

        Ok(kind_slugs)
    }

    /// Generates the code for a field with no attributes.
    fn generate_field_set_arg(&mut self, binding: &BindingInfo<'_>) -> TokenStream {
        let ast = binding.ast();
        assert_eq!(ast.attrs.len(), 0, "field with attribute used as diagnostic arg");

        let diag = &self.diag;
        let ident = ast.ident.as_ref().unwrap();
        quote! {
            #diag.set_arg(
                stringify!(#ident),
                #binding
            );
        }
    }

    /// Generates the necessary code for all attributes on a field.
    fn generate_field_attr_code(
        &mut self,
        binding: &BindingInfo<'_>,
        kind_stats: KindsStatistics,
    ) -> TokenStream {
        let ast = binding.ast();
        assert!(ast.attrs.len() > 0, "field without attributes generating attr code");

        // Abstract over `Vec<T>` and `Option<T>` fields using `FieldInnerTy`, which will
        // apply the generated code on each element in the `Vec` or `Option`.
        let inner_ty = FieldInnerTy::from_type(&ast.ty);
        ast.attrs
            .iter()
            .map(|attr| {
                let info = FieldInfo {
                    binding,
                    ty: inner_ty.inner_type().unwrap_or(&ast.ty),
                    span: &ast.span(),
                };

                let generated = self
                    .generate_field_code_inner(kind_stats, attr, info)
                    .unwrap_or_else(|v| v.to_compile_error());

                inner_ty.with(binding, generated)
            })
            .collect()
    }

    fn generate_field_code_inner(
        &mut self,
        kind_stats: KindsStatistics,
        attr: &Attribute,
        info: FieldInfo<'_>,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        let meta = attr.parse_meta()?;
        match meta {
            Meta::Path(path) => self.generate_field_code_inner_path(kind_stats, attr, info, path),
            Meta::List(list @ MetaList { .. }) => {
                self.generate_field_code_inner_list(kind_stats, attr, info, list)
            }
            _ => throw_invalid_attr!(attr, &meta),
        }
    }

    /// Generates the code for a `[Meta::Path]`-like attribute on a field (e.g. `#[primary_span]`).
    fn generate_field_code_inner_path(
        &mut self,
        kind_stats: KindsStatistics,
        attr: &Attribute,
        info: FieldInfo<'_>,
        path: Path,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        let span = attr.span().unwrap();
        let ident = &path.segments.last().unwrap().ident;
        let name = ident.to_string();
        let name = name.as_str();

        match name {
            "skip_arg" => Ok(quote! {}),
            "primary_span" => {
                if kind_stats.has_multipart_suggestion {
                    throw_invalid_attr!(attr, &Meta::Path(path), |diag| {
                        diag.help(
                            "multipart suggestions use one or more `#[suggestion_part]`s rather \
                            than one `#[primary_span]`",
                        )
                    })
                }

                report_error_if_not_applied_to_span(attr, &info)?;

                let binding = info.binding.binding.clone();
                self.span_field.set_once((binding, span));

                Ok(quote! {})
            }
            "suggestion_part" => {
                self.has_suggestion_parts = true;

                if kind_stats.has_multipart_suggestion {
                    span_err(span, "`#[suggestion_part(...)]` attribute without `code = \"...\"`")
                        .emit();
                    Ok(quote! {})
                } else {
                    throw_invalid_attr!(attr, &Meta::Path(path), |diag| {
                        diag.help(
                                "`#[suggestion_part(...)]` is only valid in multipart suggestions, use `#[primary_span]` instead",
                            )
                    });
                }
            }
            "applicability" => {
                if kind_stats.has_multipart_suggestion || kind_stats.has_normal_suggestion {
                    report_error_if_not_applied_to_applicability(attr, &info)?;

                    let binding = info.binding.binding.clone();
                    self.applicability.set_once((quote! { #binding }, span));
                } else {
                    span_err(span, "`#[applicability]` is only valid on suggestions").emit();
                }

                Ok(quote! {})
            }
            _ => throw_invalid_attr!(attr, &Meta::Path(path), |diag| {
                let mut span_attrs = vec![];
                if kind_stats.has_multipart_suggestion {
                    span_attrs.push("suggestion_part");
                }
                if !kind_stats.all_multipart_suggestions {
                    span_attrs.push("primary_span")
                }
                diag.help(format!(
                    "only `{}`, `applicability` and `skip_arg` are valid field attributes",
                    span_attrs.join(", ")
                ))
            }),
        }
    }

    /// Generates the code for a `[Meta::List]`-like attribute on a field (e.g.
    /// `#[suggestion_part(code = "...")]`).
    fn generate_field_code_inner_list(
        &mut self,
        kind_stats: KindsStatistics,
        attr: &Attribute,
        info: FieldInfo<'_>,
        list: MetaList,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        let span = attr.span().unwrap();
        let ident = &list.path.segments.last().unwrap().ident;
        let name = ident.to_string();
        let name = name.as_str();

        match name {
            "suggestion_part" => {
                if !kind_stats.has_multipart_suggestion {
                    throw_invalid_attr!(attr, &Meta::List(list), |diag| {
                        diag.help(
                            "`#[suggestion_part(...)]` is only valid in multipart suggestions",
                        )
                    })
                }

                self.has_suggestion_parts = true;

                report_error_if_not_applied_to_span(attr, &info)?;

                let mut code = None;
                for nested_attr in list.nested.iter() {
                    let NestedMeta::Meta(ref meta) = nested_attr else {
                        throw_invalid_nested_attr!(attr, &nested_attr);
                    };

                    let span = meta.span().unwrap();
                    let nested_name = meta.path().segments.last().unwrap().ident.to_string();
                    let nested_name = nested_name.as_str();

                    let Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(value), .. }) = meta else {
                        throw_invalid_nested_attr!(attr, &nested_attr);
                    };

                    match nested_name {
                        "code" => {
                            let formatted_str = self.build_format(&value.value(), value.span());
                            code.set_once((formatted_str, span));
                        }
                        _ => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                            diag.help("`code` is the only valid nested attribute")
                        }),
                    }
                }

                let Some((code, _)) = code else {
                    span_err(span, "`#[suggestion_part(...)]` attribute without `code = \"...\"`")
                        .emit();
                    return Ok(quote! {});
                };
                let binding = info.binding;

                Ok(quote! { suggestions.push((#binding, #code)); })
            }
            _ => throw_invalid_attr!(attr, &Meta::List(list), |diag| {
                let mut span_attrs = vec![];
                if kind_stats.has_multipart_suggestion {
                    span_attrs.push("suggestion_part");
                }
                if !kind_stats.all_multipart_suggestions {
                    span_attrs.push("primary_span")
                }
                diag.help(format!(
                    "only `{}`, `applicability` and `skip_arg` are valid field attributes",
                    span_attrs.join(", ")
                ))
            }),
        }
    }

    pub fn into_tokens(&mut self) -> Result<TokenStream, DiagnosticDeriveError> {
        let kind_slugs = self.identify_kind()?;
        if kind_slugs.is_empty() {
            throw_span_err!(
                self.variant.ast().ident.span().unwrap(),
                "subdiagnostic kind not specified"
            );
        };

        let kind_stats: KindsStatistics = kind_slugs.iter().map(|(kind, _slug)| kind).collect();

        let init = if kind_stats.has_multipart_suggestion {
            quote! { let mut suggestions = Vec::new(); }
        } else {
            quote! {}
        };

        let attr_args: TokenStream = self
            .variant
            .bindings()
            .iter()
            .filter(|binding| !binding.ast().attrs.is_empty())
            .map(|binding| self.generate_field_attr_code(binding, kind_stats))
            .collect();

        let span_field = self.span_field.as_ref().map(|(span, _)| span);
        let applicability = self.applicability.take().map_or_else(
            || quote! { rustc_errors::Applicability::Unspecified },
            |(applicability, _)| applicability,
        );

        let diag = &self.diag;
        let mut calls = TokenStream::new();
        for (kind, slug) in kind_slugs {
            let name = format_ident!("{}{}", if span_field.is_some() { "span_" } else { "" }, kind);
            let message = quote! { rustc_errors::fluent::#slug };
            let call = match kind {
                SubdiagnosticKind::Suggestion { suggestion_kind, code } => {
                    if let Some(span) = span_field {
                        let style = suggestion_kind.to_suggestion_style();

                        quote! { #diag.#name(#span, #message, #code, #applicability, #style); }
                    } else {
                        span_err(self.span, "suggestion without `#[primary_span]` field").emit();
                        quote! { unreachable!(); }
                    }
                }
                SubdiagnosticKind::MultipartSuggestion { suggestion_kind } => {
                    if !self.has_suggestion_parts {
                        span_err(
                            self.span,
                            "multipart suggestion without any `#[suggestion_part(...)]` fields",
                        )
                        .emit();
                    }

                    let style = suggestion_kind.to_suggestion_style();

                    quote! { #diag.#name(#message, suggestions, #applicability, #style); }
                }
                SubdiagnosticKind::Label => {
                    if let Some(span) = span_field {
                        quote! { #diag.#name(#span, #message); }
                    } else {
                        span_err(self.span, "label without `#[primary_span]` field").emit();
                        quote! { unreachable!(); }
                    }
                }
                _ => {
                    if let Some(span) = span_field {
                        quote! { #diag.#name(#span, #message); }
                    } else {
                        quote! { #diag.#name(#message); }
                    }
                }
            };
            calls.extend(call);
        }

        let plain_args: TokenStream = self
            .variant
            .bindings()
            .iter()
            .filter(|binding| binding.ast().attrs.is_empty())
            .map(|binding| self.generate_field_set_arg(binding))
            .collect();

        Ok(quote! {
            #init
            #attr_args
            #calls
            #plain_args
        })
    }
}
