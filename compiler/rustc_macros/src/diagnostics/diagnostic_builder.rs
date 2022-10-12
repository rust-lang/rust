#![deny(unused_must_use)]

use crate::diagnostics::error::{
    invalid_nested_attr, span_err, throw_invalid_attr, throw_invalid_nested_attr, throw_span_err,
    DiagnosticDeriveError,
};
use crate::diagnostics::utils::{
    bind_style_of_field, build_field_mapping, report_error_if_not_applied_to_span,
    report_type_error, should_generate_set_arg, type_is_unit, type_matches_path, FieldInfo,
    FieldInnerTy, FieldMap, HasFieldMap, SetOnce, SpannedOption, SubdiagnosticKind,
};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    parse_quote, spanned::Spanned, Attribute, Meta, MetaList, MetaNameValue, NestedMeta, Path, Type,
};
use synstructure::{BindingInfo, Structure, VariantInfo};

/// What kind of diagnostic is being derived - a fatal/error/warning or a lint?
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) enum DiagnosticDeriveKind {
    Diagnostic,
    LintDiagnostic,
}

/// Tracks persistent information required for the entire type when building up individual calls to
/// diagnostic methods for generated diagnostic derives - both `Diagnostic` for
/// fatal/errors/warnings and `LintDiagnostic` for lints.
pub(crate) struct DiagnosticDeriveBuilder {
    /// The identifier to use for the generated `DiagnosticBuilder` instance.
    pub diag: syn::Ident,
    /// Kind of diagnostic that should be derived.
    pub kind: DiagnosticDeriveKind,
}

/// Tracks persistent information required for a specific variant when building up individual calls
/// to diagnostic methods for generated diagnostic derives - both `Diagnostic` for
/// fatal/errors/warnings and `LintDiagnostic` for lints.
pub(crate) struct DiagnosticDeriveVariantBuilder<'parent> {
    /// The parent builder for the entire type.
    pub parent: &'parent DiagnosticDeriveBuilder,

    /// Span of the struct or the enum variant.
    pub span: proc_macro::Span,

    /// Store a map of field name to its corresponding field. This is built on construction of the
    /// derive builder.
    pub field_map: FieldMap,

    /// Slug is a mandatory part of the struct attribute as corresponds to the Fluent message that
    /// has the actual diagnostic message.
    pub slug: SpannedOption<Path>,
    /// Error codes are a optional part of the struct attribute - this is only set to detect
    /// multiple specifications.
    pub code: SpannedOption<()>,
}

impl<'a> HasFieldMap for DiagnosticDeriveVariantBuilder<'a> {
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream> {
        self.field_map.get(field)
    }
}

impl DiagnosticDeriveBuilder {
    /// Call `f` for the struct or for each variant of the enum, returning a `TokenStream` with the
    /// tokens from `f` wrapped in an `match` expression. Emits errors for use of derive on unions
    /// or attributes on the type itself when input is an enum.
    pub fn each_variant<'s, F>(&mut self, structure: &mut Structure<'s>, f: F) -> TokenStream
    where
        F: for<'a, 'v> Fn(DiagnosticDeriveVariantBuilder<'a>, &VariantInfo<'v>) -> TokenStream,
    {
        let ast = structure.ast();
        let span = ast.span().unwrap();
        match ast.data {
            syn::Data::Struct(..) | syn::Data::Enum(..) => (),
            syn::Data::Union(..) => {
                span_err(span, "diagnostic derives can only be used on structs and enums");
            }
        }

        if matches!(ast.data, syn::Data::Enum(..)) {
            for attr in &ast.attrs {
                span_err(
                    attr.span().unwrap(),
                    "unsupported type attribute for diagnostic derive enum",
                )
                .emit();
            }
        }

        for variant in structure.variants_mut() {
            // First, change the binding style of each field based on the code that will be
            // generated for the field - e.g. `set_arg` calls needs by-move bindings, whereas
            // `set_primary_span` only needs by-ref.
            variant.bind_with(|bi| bind_style_of_field(bi.ast()).0);

            // Then, perform a stable sort on bindings which generates code for by-ref bindings
            // before code generated for by-move bindings. Any code generated for the by-ref
            // bindings which creates a reference to the by-move fields will happen before the
            // by-move bindings move those fields and make them inaccessible.
            variant.bindings_mut().sort_by_cached_key(|bi| bind_style_of_field(bi.ast()));
        }

        let variants = structure.each_variant(|variant| {
            let span = match structure.ast().data {
                syn::Data::Struct(..) => span,
                // There isn't a good way to get the span of the variant, so the variant's
                // name will need to do.
                _ => variant.ast().ident.span().unwrap(),
            };
            let builder = DiagnosticDeriveVariantBuilder {
                parent: &self,
                span,
                field_map: build_field_mapping(variant),
                slug: None,
                code: None,
            };
            f(builder, variant)
        });

        quote! {
            match self {
                #variants
            }
        }
    }
}

impl<'a> DiagnosticDeriveVariantBuilder<'a> {
    /// Generates calls to `code` and similar functions based on the attributes on the type or
    /// variant.
    pub fn preamble<'s>(&mut self, variant: &VariantInfo<'s>) -> TokenStream {
        let ast = variant.ast();
        let attrs = &ast.attrs;
        let preamble = attrs.iter().map(|attr| {
            self.generate_structure_code_for_attr(attr).unwrap_or_else(|v| v.to_compile_error())
        });

        quote! {
            #(#preamble)*;
        }
    }

    /// Generates calls to `span_label` and similar functions based on the attributes on fields or
    /// calls to `set_arg` when no attributes are present.
    ///
    /// Expects use of `Self::each_variant` which will have sorted bindings so that by-ref bindings
    /// (which may create references to by-move bindings) have their code generated first -
    /// necessary as code for suggestions uses formatting machinery and the value of other fields
    /// (any given field can be referenced multiple times, so must be accessed through a borrow);
    /// and when passing fields to `add_subdiagnostic` or `set_arg` for Fluent, fields must be
    /// accessed by-move.
    pub fn body<'s>(&mut self, variant: &VariantInfo<'s>) -> TokenStream {
        let mut body = quote! {};
        for binding in variant.bindings() {
            body.extend(self.generate_field_attrs_code(binding));
        }
        body
    }

    /// Parse a `SubdiagnosticKind` from an `Attribute`.
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
        let diag = &self.parent.diag;

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

        if should_generate_set_arg(&field) {
            let diag = &self.parent.diag;
            let ident = field.ident.as_ref().unwrap();
            // strip `r#` prefix, if present
            let ident = format_ident!("{}", ident);
            return quote! {
                #diag.set_arg(
                    stringify!(#ident),
                    #field_binding
                );
            };
        }

        let needs_move = bind_style_of_field(&field).is_move();
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
        let diag = &self.parent.diag;
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
                "primary_span" => match self.parent.kind {
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
        let diag = &self.parent.diag;
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
        let diag = &self.parent.diag;
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
