#![deny(unused_must_use)]

use crate::diagnostics::error::{
    invalid_attr, span_err, throw_invalid_attr, throw_span_err, DiagnosticDeriveError,
};
use crate::diagnostics::utils::{
    build_field_mapping, build_suggestion_code, is_doc_comment, new_code_ident,
    report_error_if_not_applied_to_applicability, report_error_if_not_applied_to_span,
    should_generate_set_arg, AllowMultipleAlternatives, FieldInfo, FieldInnerTy, FieldMap,
    HasFieldMap, SetOnce, SpannedOption, SubdiagnosticKind,
};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{spanned::Spanned, Attribute, Meta, MetaList, Path};
use synstructure::{BindingInfo, Structure, VariantInfo};

use super::utils::SubdiagnosticVariant;

/// The central struct for constructing the `add_to_diagnostic` method from an annotated struct.
pub(crate) struct SubdiagnosticDeriveBuilder {
    diag: syn::Ident,
    f: syn::Ident,
}

impl SubdiagnosticDeriveBuilder {
    pub(crate) fn new() -> Self {
        let diag = format_ident!("diag");
        let f = format_ident!("f");
        Self { diag, f }
    }

    pub(crate) fn into_tokens(self, mut structure: Structure<'_>) -> TokenStream {
        let implementation = {
            let ast = structure.ast();
            let span = ast.span().unwrap();
            match ast.data {
                syn::Data::Struct(..) | syn::Data::Enum(..) => (),
                syn::Data::Union(..) => {
                    span_err(
                        span,
                        "`#[derive(Subdiagnostic)]` can only be used on structs and enums",
                    )
                    .emit();
                }
            }

            let is_enum = matches!(ast.data, syn::Data::Enum(..));
            if is_enum {
                for attr in &ast.attrs {
                    // Always allow documentation comments.
                    if is_doc_comment(attr) {
                        continue;
                    }

                    span_err(
                        attr.span().unwrap(),
                        "unsupported type attribute for subdiagnostic enum",
                    )
                    .emit();
                }
            }

            structure.bind_with(|_| synstructure::BindStyle::Move);
            let variants_ = structure.each_variant(|variant| {
                let mut builder = SubdiagnosticDeriveVariantBuilder {
                    parent: &self,
                    variant,
                    span,
                    formatting_init: TokenStream::new(),
                    fields: build_field_mapping(variant),
                    span_field: None,
                    applicability: None,
                    has_suggestion_parts: false,
                    is_enum,
                };
                builder.into_tokens().unwrap_or_else(|v| v.to_compile_error())
            });

            quote! {
                match self {
                    #variants_
                }
            }
        };

        let diag = &self.diag;
        let f = &self.f;
        let ret = structure.gen_impl(quote! {
            gen impl rustc_errors::AddToDiagnostic for @Self {
                fn add_to_diagnostic_with<__F>(self, #diag: &mut rustc_errors::Diagnostic, #f: __F)
                where
                    __F: core::ops::Fn(
                        &mut rustc_errors::Diagnostic,
                        rustc_errors::SubdiagnosticMessage
                    ) -> rustc_errors::SubdiagnosticMessage,
                {
                    use rustc_errors::{Applicability, IntoDiagnosticArg};
                    #implementation
                }
            }
        });
        ret
    }
}

/// Tracks persistent information required for building up the call to add to the diagnostic
/// for the final generated method. This is a separate struct to `SubdiagnosticDerive`
/// only to be able to destructure and split `self.builder` and the `self.structure` up to avoid a
/// double mut borrow later on.
struct SubdiagnosticDeriveVariantBuilder<'parent, 'a> {
    /// The identifier to use for the generated `DiagnosticBuilder` instance.
    parent: &'parent SubdiagnosticDeriveBuilder,

    /// Info for the current variant (or the type if not an enum).
    variant: &'a VariantInfo<'a>,
    /// Span for the entire type.
    span: proc_macro::Span,

    /// Initialization of format strings for code suggestions.
    formatting_init: TokenStream,

    /// Store a map of field name to its corresponding field. This is built on construction of the
    /// derive builder.
    fields: FieldMap,

    /// Identifier for the binding to the `#[primary_span]` field.
    span_field: SpannedOption<proc_macro2::Ident>,

    /// The binding to the `#[applicability]` field, if present.
    applicability: SpannedOption<TokenStream>,

    /// Set to true when a `#[suggestion_part]` field is encountered, used to generate an error
    /// during finalization if still `false`.
    has_suggestion_parts: bool,

    /// Set to true when this variant is an enum variant rather than just the body of a struct.
    is_enum: bool,
}

impl<'parent, 'a> HasFieldMap for SubdiagnosticDeriveVariantBuilder<'parent, 'a> {
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
    all_applicabilities_static: bool,
}

impl<'a> FromIterator<&'a SubdiagnosticKind> for KindsStatistics {
    fn from_iter<T: IntoIterator<Item = &'a SubdiagnosticKind>>(kinds: T) -> Self {
        let mut ret = Self {
            has_multipart_suggestion: false,
            all_multipart_suggestions: true,
            has_normal_suggestion: false,
            all_applicabilities_static: true,
        };

        for kind in kinds {
            if let SubdiagnosticKind::MultipartSuggestion { applicability: None, .. }
            | SubdiagnosticKind::Suggestion { applicability: None, .. } = kind
            {
                ret.all_applicabilities_static = false;
            }
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

impl<'parent, 'a> SubdiagnosticDeriveVariantBuilder<'parent, 'a> {
    fn identify_kind(
        &mut self,
    ) -> Result<Vec<(SubdiagnosticKind, Path, bool)>, DiagnosticDeriveError> {
        let mut kind_slugs = vec![];

        for attr in self.variant.ast().attrs {
            let Some(SubdiagnosticVariant { kind, slug, no_span }) =
                SubdiagnosticVariant::from_attr(attr, self)?
            else {
                // Some attributes aren't errors - like documentation comments - but also aren't
                // subdiagnostics.
                continue;
            };

            let Some(slug) = slug else {
                let name = attr.path().segments.last().unwrap().ident.to_string();
                let name = name.as_str();

                throw_span_err!(
                    attr.span().unwrap(),
                    format!(
                        "diagnostic slug must be first argument of a `#[{name}(...)]` attribute"
                    )
                );
            };

            kind_slugs.push((kind, slug, no_span));
        }

        Ok(kind_slugs)
    }

    /// Generates the code for a field with no attributes.
    fn generate_field_set_arg(&mut self, binding_info: &BindingInfo<'_>) -> TokenStream {
        let diag = &self.parent.diag;

        let field = binding_info.ast();
        let mut field_binding = binding_info.binding.clone();
        field_binding.set_span(field.ty.span());

        let ident = field.ident.as_ref().unwrap();
        let ident = format_ident!("{}", ident); // strip `r#` prefix, if present

        quote! {
            #diag.set_arg(
                stringify!(#ident),
                #field_binding
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
                // Always allow documentation comments.
                if is_doc_comment(attr) {
                    return quote! {};
                }

                let info = FieldInfo { binding, ty: inner_ty, span: &ast.span() };

                let generated = self
                    .generate_field_code_inner(kind_stats, attr, info, inner_ty.will_iterate())
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
        clone_suggestion_code: bool,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        match &attr.meta {
            Meta::Path(path) => {
                self.generate_field_code_inner_path(kind_stats, attr, info, path.clone())
            }
            Meta::List(list) => self.generate_field_code_inner_list(
                kind_stats,
                attr,
                info,
                list,
                clone_suggestion_code,
            ),
            _ => throw_invalid_attr!(attr),
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
                    invalid_attr(attr)
                        .help(
                            "multipart suggestions use one or more `#[suggestion_part]`s rather \
                            than one `#[primary_span]`",
                        )
                        .emit();
                } else {
                    report_error_if_not_applied_to_span(attr, &info)?;

                    let binding = info.binding.binding.clone();
                    // FIXME(#100717): support `Option<Span>` on `primary_span` like in the
                    // diagnostic derive
                    if !matches!(info.ty, FieldInnerTy::Plain(_)) {
                        throw_invalid_attr!(attr, |diag| {
                            let diag = diag.note("there must be exactly one primary span");

                            if kind_stats.has_normal_suggestion {
                                diag.help(
                                    "to create a suggestion with multiple spans, \
                                     use `#[multipart_suggestion]` instead",
                                )
                            } else {
                                diag
                            }
                        });
                    }

                    self.span_field.set_once(binding, span);
                }

                Ok(quote! {})
            }
            "suggestion_part" => {
                self.has_suggestion_parts = true;

                if kind_stats.has_multipart_suggestion {
                    span_err(span, "`#[suggestion_part(...)]` attribute without `code = \"...\"`")
                        .emit();
                } else {
                    invalid_attr(attr)
                        .help(
                            "`#[suggestion_part(...)]` is only valid in multipart suggestions, \
                             use `#[primary_span]` instead",
                        )
                        .emit();
                }

                Ok(quote! {})
            }
            "applicability" => {
                if kind_stats.has_multipart_suggestion || kind_stats.has_normal_suggestion {
                    report_error_if_not_applied_to_applicability(attr, &info)?;

                    if kind_stats.all_applicabilities_static {
                        span_err(
                            span,
                            "`#[applicability]` has no effect if all `#[suggestion]`/\
                             `#[multipart_suggestion]` attributes have a static \
                             `applicability = \"...\"`",
                        )
                        .emit();
                    }
                    let binding = info.binding.binding.clone();
                    self.applicability.set_once(quote! { #binding }, span);
                } else {
                    span_err(span, "`#[applicability]` is only valid on suggestions").emit();
                }

                Ok(quote! {})
            }
            _ => {
                let mut span_attrs = vec![];
                if kind_stats.has_multipart_suggestion {
                    span_attrs.push("suggestion_part");
                }
                if !kind_stats.all_multipart_suggestions {
                    span_attrs.push("primary_span")
                }

                invalid_attr(attr)
                    .help(format!(
                        "only `{}`, `applicability` and `skip_arg` are valid field attributes",
                        span_attrs.join(", ")
                    ))
                    .emit();

                Ok(quote! {})
            }
        }
    }

    /// Generates the code for a `[Meta::List]`-like attribute on a field (e.g.
    /// `#[suggestion_part(code = "...")]`).
    fn generate_field_code_inner_list(
        &mut self,
        kind_stats: KindsStatistics,
        attr: &Attribute,
        info: FieldInfo<'_>,
        list: &MetaList,
        clone_suggestion_code: bool,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        let span = attr.span().unwrap();
        let mut ident = list.path.segments.last().unwrap().ident.clone();
        ident.set_span(info.ty.span());
        let name = ident.to_string();
        let name = name.as_str();

        match name {
            "suggestion_part" => {
                if !kind_stats.has_multipart_suggestion {
                    throw_invalid_attr!(attr, |diag| {
                        diag.help(
                            "`#[suggestion_part(...)]` is only valid in multipart suggestions",
                        )
                    })
                }

                self.has_suggestion_parts = true;

                report_error_if_not_applied_to_span(attr, &info)?;

                let mut code = None;

                list.parse_nested_meta(|nested| {
                    if nested.path.is_ident("code") {
                        let code_field = new_code_ident();
                        let span = nested.path.span().unwrap();
                        let formatting_init = build_suggestion_code(
                            &code_field,
                            nested,
                            self,
                            AllowMultipleAlternatives::No,
                        );
                        code.set_once((code_field, formatting_init), span);
                    } else {
                        span_err(
                            nested.path.span().unwrap(),
                            "`code` is the only valid nested attribute",
                        )
                        .emit();
                    }
                    Ok(())
                })?;

                let Some((code_field, formatting_init)) = code.value() else {
                    span_err(span, "`#[suggestion_part(...)]` attribute without `code = \"...\"`")
                        .emit();
                    return Ok(quote! {});
                };
                let binding = info.binding;

                self.formatting_init.extend(formatting_init);
                let code_field = if clone_suggestion_code {
                    quote! { #code_field.clone() }
                } else {
                    quote! { #code_field }
                };
                Ok(quote! { suggestions.push((#binding, #code_field)); })
            }
            _ => throw_invalid_attr!(attr, |diag| {
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
            if self.is_enum {
                // It's okay for a variant to not be a subdiagnostic at all..
                return Ok(quote! {});
            } else {
                // ..but structs should always be _something_.
                throw_span_err!(
                    self.variant.ast().ident.span().unwrap(),
                    "subdiagnostic kind not specified"
                );
            }
        };

        let kind_stats: KindsStatistics =
            kind_slugs.iter().map(|(kind, _slug, _no_span)| kind).collect();

        let init = if kind_stats.has_multipart_suggestion {
            quote! { let mut suggestions = Vec::new(); }
        } else {
            quote! {}
        };

        let attr_args: TokenStream = self
            .variant
            .bindings()
            .iter()
            .filter(|binding| !should_generate_set_arg(binding.ast()))
            .map(|binding| self.generate_field_attr_code(binding, kind_stats))
            .collect();

        let span_field = self.span_field.value_ref();

        let diag = &self.parent.diag;
        let f = &self.parent.f;
        let mut calls = TokenStream::new();
        for (kind, slug, no_span) in kind_slugs {
            let message = format_ident!("__message");
            calls.extend(
                quote! { let #message = #f(#diag, crate::fluent_generated::#slug.into()); },
            );

            let name = format_ident!(
                "{}{}",
                if span_field.is_some() && !no_span { "span_" } else { "" },
                kind
            );
            let call = match kind {
                SubdiagnosticKind::Suggestion {
                    suggestion_kind,
                    applicability,
                    code_init,
                    code_field,
                } => {
                    self.formatting_init.extend(code_init);

                    let applicability = applicability
                        .value()
                        .map(|a| quote! { #a })
                        .or_else(|| self.applicability.take().value())
                        .unwrap_or_else(|| quote! { rustc_errors::Applicability::Unspecified });

                    if let Some(span) = span_field {
                        let style = suggestion_kind.to_suggestion_style();
                        quote! { #diag.#name(#span, #message, #code_field, #applicability, #style); }
                    } else {
                        span_err(self.span, "suggestion without `#[primary_span]` field").emit();
                        quote! { unreachable!(); }
                    }
                }
                SubdiagnosticKind::MultipartSuggestion { suggestion_kind, applicability } => {
                    let applicability = applicability
                        .value()
                        .map(|a| quote! { #a })
                        .or_else(|| self.applicability.take().value())
                        .unwrap_or_else(|| quote! { rustc_errors::Applicability::Unspecified });

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
                    if let Some(span) = span_field && !no_span {
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
            .filter(|binding| should_generate_set_arg(binding.ast()))
            .map(|binding| self.generate_field_set_arg(binding))
            .collect();

        let formatting_init = &self.formatting_init;
        Ok(quote! {
            #init
            #formatting_init
            #attr_args
            #plain_args
            #calls
        })
    }
}
