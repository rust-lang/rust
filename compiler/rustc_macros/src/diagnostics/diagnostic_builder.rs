#![deny(unused_must_use)]

use super::error::throw_invalid_nested_attr;
use super::utils::{SpannedOption, SubdiagnosticKind};
use crate::diagnostics::error::{
    invalid_nested_attr, span_err, throw_invalid_attr, throw_span_err, DiagnosticDeriveError,
};
use crate::diagnostics::utils::{
    report_error_if_not_applied_to_span, report_type_error, type_is_unit, type_matches_path,
    FieldInfo, FieldInnerTy, HasFieldMap, SetOnce,
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use std::collections::HashMap;
use syn::{
    parse_quote, spanned::Spanned, Attribute, Field, Meta, MetaList, MetaNameValue, NestedMeta,
    Path, Type,
};
use synstructure::{BindingInfo, Structure};

/// What kind of diagnostic is being derived - a fatal/error/warning or a lint?
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum DiagnosticDeriveKind {
    Diagnostic,
    LintDiagnostic,
}

/// Tracks persistent information required for building up individual calls to diagnostic methods
/// for generated diagnostic derives - both `Diagnostic` for fatal/errors/warnings and
/// `LintDiagnostic` for lints.
pub(crate) struct DiagnosticDeriveBuilder {
    /// The identifier to use for the generated `DiagnosticBuilder` instance.
    pub diag: syn::Ident,

    /// Store a map of field name to its corresponding field. This is built on construction of the
    /// derive builder.
    pub fields: HashMap<String, TokenStream>,

    /// Kind of diagnostic that should be derived.
    pub kind: DiagnosticDeriveKind,
    /// Slug is a mandatory part of the struct attribute as corresponds to the Fluent message that
    /// has the actual diagnostic message.
    pub slug: SpannedOption<Path>,
    /// Error codes are a optional part of the struct attribute - this is only set to detect
    /// multiple specifications.
    pub code: SpannedOption<()>,
}

impl HasFieldMap for DiagnosticDeriveBuilder {
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream> {
        self.fields.get(field)
    }
}

impl DiagnosticDeriveBuilder {
    pub fn preamble<'s>(&mut self, structure: &Structure<'s>) -> TokenStream {
        let ast = structure.ast();
        let attrs = &ast.attrs;
        let preamble = attrs.iter().map(|attr| {
            self.generate_structure_code_for_attr(attr).unwrap_or_else(|v| v.to_compile_error())
        });

        quote! {
            #(#preamble)*;
        }
    }

    pub fn body<'s>(&mut self, structure: &mut Structure<'s>) -> (TokenStream, TokenStream) {
        // Keep track of which fields need to be handled with a by-move binding.
        let mut needs_moved = std::collections::HashSet::new();

        // Generates calls to `span_label` and similar functions based on the attributes
        // on fields. Code for suggestions uses formatting machinery and the value of
        // other fields - because any given field can be referenced multiple times, it
        // should be accessed through a borrow. When passing fields to `add_subdiagnostic`
        // or `set_arg` (which happens below) for Fluent, we want to move the data, so that
        // has to happen in a separate pass over the fields.
        let attrs = structure
            .clone()
            .filter(|field_binding| {
                let ast = &field_binding.ast();
                !self.needs_move(ast) || {
                    needs_moved.insert(field_binding.binding.clone());
                    false
                }
            })
            .each(|field_binding| self.generate_field_attrs_code(field_binding));

        structure.bind_with(|_| synstructure::BindStyle::Move);
        // When a field has attributes like `#[label]` or `#[note]` then it doesn't
        // need to be passed as an argument to the diagnostic. But when a field has no
        // attributes or a `#[subdiagnostic]` attribute then it must be passed as an
        // argument to the diagnostic so that it can be referred to by Fluent messages.
        let args = structure
            .filter(|field_binding| needs_moved.contains(&field_binding.binding))
            .each(|field_binding| self.generate_field_attrs_code(field_binding));

        (attrs, args)
    }

    /// Returns `true` if `field` should generate a `set_arg` call rather than any other diagnostic
    /// call (like `span_label`).
    fn should_generate_set_arg(&self, field: &Field) -> bool {
        field.attrs.is_empty()
    }

    /// Returns `true` if `field` needs to have code generated in the by-move branch of the
    /// generated derive rather than the by-ref branch.
    fn needs_move(&self, field: &Field) -> bool {
        let generates_set_arg = self.should_generate_set_arg(field);
        let is_multispan = type_matches_path(&field.ty, &["rustc_errors", "MultiSpan"]);
        // FIXME(davidtwco): better support for one field needing to be in the by-move and
        // by-ref branches.
        let is_subdiagnostic = field
            .attrs
            .iter()
            .map(|attr| attr.path.segments.last().unwrap().ident.to_string())
            .any(|attr| attr == "subdiagnostic");

        // `set_arg` calls take their argument by-move..
        generates_set_arg
            // If this is a `MultiSpan` field then it needs to be moved to be used by any
            // attribute..
            || is_multispan
            // If this a `#[subdiagnostic]` then it needs to be moved as the other diagnostic is
            // unlikely to be `Copy`..
            || is_subdiagnostic
    }

    fn parse_subdiag_attribute(
        &self,
        attr: &Attribute,
    ) -> Result<(SubdiagnosticKind, Path), DiagnosticDeriveError> {
        let (subdiag, slug) = SubdiagnosticKind::from_attr(attr, self)?;

        if let SubdiagnosticKind::MultipartSuggestion { .. } = subdiag {
            let meta = attr.parse_meta()?;
            throw_invalid_attr!(attr, &meta, |diag| diag
                .help("consider creating a `Subdiagnostic` instead"));
        }

        let slug = slug.unwrap_or_else(|| match subdiag {
            SubdiagnosticKind::Label => parse_quote! { _subdiag::label },
            SubdiagnosticKind::Note => parse_quote! { _subdiag::note },
            SubdiagnosticKind::Help => parse_quote! { _subdiag::help },
            SubdiagnosticKind::Warn => parse_quote! { _subdiag::warn },
            SubdiagnosticKind::Suggestion { .. } => parse_quote! { _subdiag::suggestion },
            SubdiagnosticKind::MultipartSuggestion { .. } => unreachable!(),
        });

        Ok((subdiag, slug))
    }

    /// Establishes state in the `DiagnosticDeriveBuilder` resulting from the struct
    /// attributes like `#[diag(..)]`, such as the slug and error code. Generates
    /// diagnostic builder calls for setting error code and creating note/help messages.
    fn generate_structure_code_for_attr(
        &mut self,
        attr: &Attribute,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        let diag = &self.diag;

        let name = attr.path.segments.last().unwrap().ident.to_string();
        let name = name.as_str();
        let meta = attr.parse_meta()?;

        if name == "diag" {
            let Meta::List(MetaList { ref nested, .. }) = meta else {
                throw_invalid_attr!(
                    attr,
                    &meta
                );
            };

            let mut nested_iter = nested.into_iter().peekable();

            match nested_iter.peek() {
                Some(NestedMeta::Meta(Meta::Path(slug))) => {
                    self.slug.set_once(slug.clone(), slug.span().unwrap());
                    nested_iter.next();
                }
                Some(NestedMeta::Meta(Meta::NameValue { .. })) => {}
                Some(nested_attr) => throw_invalid_nested_attr!(attr, &nested_attr, |diag| diag
                    .help("a diagnostic slug is required as the first argument")),
                None => throw_invalid_attr!(attr, &meta, |diag| diag
                    .help("a diagnostic slug is required as the first argument")),
            };

            // Remaining attributes are optional, only `code = ".."` at the moment.
            let mut tokens = TokenStream::new();
            for nested_attr in nested_iter {
                let (value, path) = match nested_attr {
                    NestedMeta::Meta(Meta::NameValue(MetaNameValue {
                        lit: syn::Lit::Str(value),
                        path,
                        ..
                    })) => (value, path),
                    NestedMeta::Meta(Meta::Path(_)) => {
                        invalid_nested_attr(attr, &nested_attr)
                            .help("diagnostic slug must be the first argument")
                            .emit();
                        continue;
                    }
                    _ => {
                        invalid_nested_attr(attr, &nested_attr).emit();
                        continue;
                    }
                };

                let nested_name = path.segments.last().unwrap().ident.to_string();
                // Struct attributes are only allowed to be applied once, and the diagnostic
                // changes will be set in the initialisation code.
                let span = value.span().unwrap();
                match nested_name.as_str() {
                    "code" => {
                        self.code.set_once((), span);

                        let code = value.value();
                        tokens.extend(quote! {
                            #diag.code(rustc_errors::DiagnosticId::Error(#code.to_string()));
                        });
                    }
                    _ => invalid_nested_attr(attr, &nested_attr)
                        .help("only `code` is a valid nested attributes following the slug")
                        .emit(),
                }
            }
            return Ok(tokens);
        }

        let (subdiag, slug) = self.parse_subdiag_attribute(attr)?;
        let fn_ident = format_ident!("{}", subdiag);
        match subdiag {
            SubdiagnosticKind::Note | SubdiagnosticKind::Help | SubdiagnosticKind::Warn => {
                Ok(self.add_subdiagnostic(&fn_ident, slug))
            }
            SubdiagnosticKind::Label | SubdiagnosticKind::Suggestion { .. } => {
                throw_invalid_attr!(attr, &meta, |diag| diag
                    .help("`#[label]` and `#[suggestion]` can only be applied to fields"));
            }
            SubdiagnosticKind::MultipartSuggestion { .. } => unreachable!(),
        }
    }

    fn generate_field_attrs_code(&mut self, binding_info: &BindingInfo<'_>) -> TokenStream {
        let field = binding_info.ast();
        let field_binding = &binding_info.binding;

        if self.should_generate_set_arg(&field) {
            let diag = &self.diag;
            let ident = field.ident.as_ref().unwrap();
            return quote! {
                #diag.set_arg(
                    stringify!(#ident),
                    #field_binding
                );
            };
        }

        let needs_move = self.needs_move(&field);
        let inner_ty = FieldInnerTy::from_type(&field.ty);

        field
            .attrs
            .iter()
            .map(move |attr| {
                let name = attr.path.segments.last().unwrap().ident.to_string();
                let needs_clone =
                    name == "primary_span" && matches!(inner_ty, FieldInnerTy::Vec(_));
                let (binding, needs_destructure) = if needs_clone {
                    // `primary_span` can accept a `Vec<Span>` so don't destructure that.
                    (quote! { #field_binding.clone() }, false)
                } else if needs_move {
                    (quote! { #field_binding }, true)
                } else {
                    (quote! { *#field_binding }, true)
                };

                let generated_code = self
                    .generate_inner_field_code(
                        attr,
                        FieldInfo {
                            binding: binding_info,
                            ty: inner_ty.inner_type().unwrap_or(&field.ty),
                            span: &field.span(),
                        },
                        binding,
                    )
                    .unwrap_or_else(|v| v.to_compile_error());

                if needs_destructure {
                    inner_ty.with(field_binding, generated_code)
                } else {
                    generated_code
                }
            })
            .collect()
    }

    fn generate_inner_field_code(
        &mut self,
        attr: &Attribute,
        info: FieldInfo<'_>,
        binding: TokenStream,
    ) -> Result<TokenStream, DiagnosticDeriveError> {
        let diag = &self.diag;
        let meta = attr.parse_meta()?;

        if let Meta::Path(_) = meta {
            let ident = &attr.path.segments.last().unwrap().ident;
            let name = ident.to_string();
            let name = name.as_str();
            match name {
                "skip_arg" => {
                    // Don't need to do anything - by virtue of the attribute existing, the
                    // `set_arg` call will not be generated.
                    return Ok(quote! {});
                }
                "primary_span" => match self.kind {
                    DiagnosticDeriveKind::Diagnostic => {
                        report_error_if_not_applied_to_span(attr, &info)?;

                        return Ok(quote! {
                            #diag.set_span(#binding);
                        });
                    }
                    DiagnosticDeriveKind::LintDiagnostic => {
                        throw_invalid_attr!(attr, &meta, |diag| {
                            diag.help("the `primary_span` field attribute is not valid for lint diagnostics")
                        })
                    }
                },
                "subdiagnostic" => return Ok(quote! { #diag.subdiagnostic(#binding); }),
                _ => {}
            }
        }

        let (subdiag, slug) = self.parse_subdiag_attribute(attr)?;

        let fn_ident = format_ident!("{}", subdiag);
        match subdiag {
            SubdiagnosticKind::Label => {
                report_error_if_not_applied_to_span(attr, &info)?;
                Ok(self.add_spanned_subdiagnostic(binding, &fn_ident, slug))
            }
            SubdiagnosticKind::Note | SubdiagnosticKind::Help | SubdiagnosticKind::Warn => {
                if type_matches_path(&info.ty, &["rustc_span", "Span"]) {
                    Ok(self.add_spanned_subdiagnostic(binding, &fn_ident, slug))
                } else if type_is_unit(&info.ty) {
                    Ok(self.add_subdiagnostic(&fn_ident, slug))
                } else {
                    report_type_error(attr, "`Span` or `()`")?
                }
            }
            SubdiagnosticKind::Suggestion {
                suggestion_kind,
                applicability: static_applicability,
                code,
            } => {
                let (span_field, mut applicability) = self.span_and_applicability_of_ty(info)?;

                if let Some((static_applicability, span)) = static_applicability {
                    applicability.set_once(quote! { #static_applicability }, span);
                }

                let applicability = applicability
                    .value()
                    .unwrap_or_else(|| quote! { rustc_errors::Applicability::Unspecified });
                let style = suggestion_kind.to_suggestion_style();

                Ok(quote! {
                    #diag.span_suggestion_with_style(
                        #span_field,
                        rustc_errors::fluent::#slug,
                        #code,
                        #applicability,
                        #style
                    );
                })
            }
            SubdiagnosticKind::MultipartSuggestion { .. } => unreachable!(),
        }
    }

    /// Adds a spanned subdiagnostic by generating a `diag.span_$kind` call with the current slug
    /// and `fluent_attr_identifier`.
    fn add_spanned_subdiagnostic(
        &self,
        field_binding: TokenStream,
        kind: &Ident,
        fluent_attr_identifier: Path,
    ) -> TokenStream {
        let diag = &self.diag;
        let fn_name = format_ident!("span_{}", kind);
        quote! {
            #diag.#fn_name(
                #field_binding,
                rustc_errors::fluent::#fluent_attr_identifier
            );
        }
    }

    /// Adds a subdiagnostic by generating a `diag.span_$kind` call with the current slug
    /// and `fluent_attr_identifier`.
    fn add_subdiagnostic(&self, kind: &Ident, fluent_attr_identifier: Path) -> TokenStream {
        let diag = &self.diag;
        quote! {
            #diag.#kind(rustc_errors::fluent::#fluent_attr_identifier);
        }
    }

    fn span_and_applicability_of_ty(
        &self,
        info: FieldInfo<'_>,
    ) -> Result<(TokenStream, SpannedOption<TokenStream>), DiagnosticDeriveError> {
        match &info.ty {
            // If `ty` is `Span` w/out applicability, then use `Applicability::Unspecified`.
            ty @ Type::Path(..) if type_matches_path(ty, &["rustc_span", "Span"]) => {
                let binding = &info.binding.binding;
                Ok((quote!(*#binding), None))
            }
            // If `ty` is `(Span, Applicability)` then return tokens accessing those.
            Type::Tuple(tup) => {
                let mut span_idx = None;
                let mut applicability_idx = None;

                fn type_err(span: &Span) -> Result<!, DiagnosticDeriveError> {
                    span_err(span.unwrap(), "wrong types for suggestion")
                        .help(
                            "`#[suggestion(...)]` on a tuple field must be applied to fields \
                             of type `(Span, Applicability)`",
                        )
                        .emit();
                    Err(DiagnosticDeriveError::ErrorHandled)
                }

                for (idx, elem) in tup.elems.iter().enumerate() {
                    if type_matches_path(elem, &["rustc_span", "Span"]) {
                        span_idx.set_once(syn::Index::from(idx), elem.span().unwrap());
                    } else if type_matches_path(elem, &["rustc_errors", "Applicability"]) {
                        applicability_idx.set_once(syn::Index::from(idx), elem.span().unwrap());
                    } else {
                        type_err(&elem.span())?;
                    }
                }

                let Some((span_idx, _)) = span_idx else {
                    type_err(&tup.span())?;
                };
                let Some((applicability_idx, applicability_span)) = applicability_idx else {
                    type_err(&tup.span())?;
                };
                let binding = &info.binding.binding;
                let span = quote!(#binding.#span_idx);
                let applicability = quote!(#binding.#applicability_idx);

                Ok((span, Some((applicability, applicability_span))))
            }
            // If `ty` isn't a `Span` or `(Span, Applicability)` then emit an error.
            _ => throw_span_err!(info.span.unwrap(), "wrong field type for suggestion", |diag| {
                diag.help(
                    "`#[suggestion(...)]` should be applied to fields of type `Span` or \
                     `(Span, Applicability)`",
                )
            }),
        }
    }
}
