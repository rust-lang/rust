#![deny(unused_must_use)]

use crate::diagnostics::error::{
    invalid_nested_attr, span_err, throw_invalid_attr, throw_invalid_nested_attr, throw_span_err,
    SessionDiagnosticDeriveError,
};
use crate::diagnostics::utils::{
    build_field_mapping, report_error_if_not_applied_to_span, report_type_error, type_is_unit,
    type_matches_path, Applicability, FieldInfo, FieldInnerTy, HasFieldMap, SetOnce,
};
use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use std::collections::HashMap;
use std::str::FromStr;
use syn::{
    parse_quote, spanned::Spanned, Attribute, Meta, MetaList, MetaNameValue, NestedMeta, Path, Type,
};
use synstructure::{BindingInfo, Structure};

/// The central struct for constructing the `into_diagnostic` method from an annotated struct.
pub(crate) struct SessionDiagnosticDerive<'a> {
    structure: Structure<'a>,
    sess: syn::Ident,
    builder: SessionDiagnosticDeriveBuilder,
}

impl<'a> SessionDiagnosticDerive<'a> {
    pub(crate) fn new(diag: syn::Ident, sess: syn::Ident, structure: Structure<'a>) -> Self {
        Self {
            builder: SessionDiagnosticDeriveBuilder {
                diag,
                fields: build_field_mapping(&structure),
                kind: None,
                code: None,
                slug: None,
            },
            sess,
            structure,
        }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let SessionDiagnosticDerive { mut structure, sess, mut builder } = self;

        let ast = structure.ast();
        let attrs = &ast.attrs;

        let (implementation, param_ty) = {
            if let syn::Data::Struct(..) = ast.data {
                let preamble = {
                    let preamble = attrs.iter().map(|attr| {
                        builder
                            .generate_structure_code(attr)
                            .unwrap_or_else(|v| v.to_compile_error())
                    });

                    quote! {
                        #(#preamble)*;
                    }
                };

                // Keep track of which fields are subdiagnostics or have no attributes.
                let mut subdiagnostics_or_empty = std::collections::HashSet::new();

                // Generates calls to `span_label` and similar functions based on the attributes
                // on fields. Code for suggestions uses formatting machinery and the value of
                // other fields - because any given field can be referenced multiple times, it
                // should be accessed through a borrow. When passing fields to `add_subdiagnostic`
                // or `set_arg` (which happens below) for Fluent, we want to move the data, so that
                // has to happen in a separate pass over the fields.
                let attrs = structure
                    .clone()
                    .filter(|field_binding| {
                        let attrs = &field_binding.ast().attrs;

                        (!attrs.is_empty()
                            && attrs.iter().all(|attr| {
                                "subdiagnostic"
                                    != attr.path.segments.last().unwrap().ident.to_string()
                            }))
                            || {
                                subdiagnostics_or_empty.insert(field_binding.binding.clone());
                                false
                            }
                    })
                    .each(|field_binding| builder.generate_field_attrs_code(field_binding));

                structure.bind_with(|_| synstructure::BindStyle::Move);
                // When a field has attributes like `#[label]` or `#[note]` then it doesn't
                // need to be passed as an argument to the diagnostic. But when a field has no
                // attributes or a `#[subdiagnostic]` attribute then it must be passed as an
                // argument to the diagnostic so that it can be referred to by Fluent messages.
                let args = structure
                    .filter(|field_binding| {
                        subdiagnostics_or_empty.contains(&field_binding.binding)
                    })
                    .each(|field_binding| builder.generate_field_attrs_code(field_binding));

                let span = ast.span().unwrap();
                let diag = &builder.diag;
                let init = match (builder.kind, builder.slug) {
                    (None, _) => {
                        span_err(span, "diagnostic kind not specified")
                            .help("use the `#[error(...)]` attribute to create an error")
                            .emit();
                        return SessionDiagnosticDeriveError::ErrorHandled.to_compile_error();
                    }
                    (Some((kind, _)), None) => {
                        span_err(span, "diagnostic slug not specified")
                            .help(&format!(
                                "specify the slug as the first argument to the attribute, such as \
                                 `#[{}(typeck::example_error)]`",
                                kind.descr()
                            ))
                            .emit();
                        return SessionDiagnosticDeriveError::ErrorHandled.to_compile_error();
                    }
                    (Some((SessionDiagnosticKind::Error, _)), Some((slug, _))) => {
                        quote! {
                            let mut #diag = #sess.struct_err(rustc_errors::fluent::#slug);
                        }
                    }
                    (Some((SessionDiagnosticKind::Warn, _)), Some((slug, _))) => {
                        quote! {
                            let mut #diag = #sess.struct_warn(rustc_errors::fluent::#slug);
                        }
                    }
                };

                let implementation = quote! {
                    #init
                    #preamble
                    match self {
                        #attrs
                    }
                    match self {
                        #args
                    }
                    #diag
                };
                let param_ty = match builder.kind {
                    Some((SessionDiagnosticKind::Error, _)) => {
                        quote! { rustc_errors::ErrorGuaranteed }
                    }
                    Some((SessionDiagnosticKind::Warn, _)) => quote! { () },
                    _ => unreachable!(),
                };

                (implementation, param_ty)
            } else {
                span_err(
                    ast.span().unwrap(),
                    "`#[derive(SessionDiagnostic)]` can only be used on structs",
                )
                .emit();

                let implementation = SessionDiagnosticDeriveError::ErrorHandled.to_compile_error();
                let param_ty = quote! { rustc_errors::ErrorGuaranteed };
                (implementation, param_ty)
            }
        };

        structure.gen_impl(quote! {
            gen impl<'__session_diagnostic_sess> rustc_session::SessionDiagnostic<'__session_diagnostic_sess, #param_ty>
                    for @Self
            {
                fn into_diagnostic(
                    self,
                    #sess: &'__session_diagnostic_sess rustc_session::parse::ParseSess
                ) -> rustc_errors::DiagnosticBuilder<'__session_diagnostic_sess, #param_ty> {
                    use rustc_errors::IntoDiagnosticArg;
                    #implementation
                }
            }
        })
    }
}

/// What kind of session diagnostic is being derived - an error or a warning?
#[derive(Copy, Clone)]
enum SessionDiagnosticKind {
    /// `#[error(..)]`
    Error,
    /// `#[warn(..)]`
    Warn,
}

impl SessionDiagnosticKind {
    /// Returns human-readable string corresponding to the kind.
    fn descr(&self) -> &'static str {
        match self {
            SessionDiagnosticKind::Error => "error",
            SessionDiagnosticKind::Warn => "warning",
        }
    }
}

/// Tracks persistent information required for building up the individual calls to diagnostic
/// methods for the final generated method. This is a separate struct to `SessionDiagnosticDerive`
/// only to be able to destructure and split `self.builder` and the `self.structure` up to avoid a
/// double mut borrow later on.
struct SessionDiagnosticDeriveBuilder {
    /// The identifier to use for the generated `DiagnosticBuilder` instance.
    diag: syn::Ident,

    /// Store a map of field name to its corresponding field. This is built on construction of the
    /// derive builder.
    fields: HashMap<String, TokenStream>,

    /// Kind of diagnostic requested via the struct attribute.
    kind: Option<(SessionDiagnosticKind, proc_macro::Span)>,
    /// Slug is a mandatory part of the struct attribute as corresponds to the Fluent message that
    /// has the actual diagnostic message.
    slug: Option<(Path, proc_macro::Span)>,
    /// Error codes are a optional part of the struct attribute - this is only set to detect
    /// multiple specifications.
    code: Option<(String, proc_macro::Span)>,
}

impl HasFieldMap for SessionDiagnosticDeriveBuilder {
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream> {
        self.fields.get(field)
    }
}

impl SessionDiagnosticDeriveBuilder {
    /// Establishes state in the `SessionDiagnosticDeriveBuilder` resulting from the struct
    /// attributes like `#[error(..)`, such as the diagnostic kind and slug. Generates
    /// diagnostic builder calls for setting error code and creating note/help messages.
    fn generate_structure_code(
        &mut self,
        attr: &Attribute,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let diag = &self.diag;
        let span = attr.span().unwrap();

        let name = attr.path.segments.last().unwrap().ident.to_string();
        let name = name.as_str();
        let meta = attr.parse_meta()?;

        let is_help_or_note = matches!(name, "help" | "note");

        let nested = match meta {
            // Most attributes are lists, like `#[error(..)]`/`#[warning(..)]` for most cases or
            // `#[help(..)]`/`#[note(..)]` when the user is specifying a alternative slug.
            Meta::List(MetaList { ref nested, .. }) => nested,
            // Subdiagnostics without spans can be applied to the type too, and these are just
            // paths: `#[help]` and `#[note]`
            Meta::Path(_) if is_help_or_note => {
                let fn_name = proc_macro2::Ident::new(name, attr.span());
                return Ok(quote! { #diag.#fn_name(rustc_errors::fluent::_subdiag::#fn_name); });
            }
            _ => throw_invalid_attr!(attr, &meta),
        };

        // Check the kind before doing any further processing so that there aren't misleading
        // "no kind specified" errors if there are failures later.
        match name {
            "error" => self.kind.set_once((SessionDiagnosticKind::Error, span)),
            "warning" => self.kind.set_once((SessionDiagnosticKind::Warn, span)),
            "help" | "note" => (),
            _ => throw_invalid_attr!(attr, &meta, |diag| {
                diag.help("only `error`, `warning`, `help` and `note` are valid attributes")
            }),
        }

        // First nested element should always be the path, e.g. `#[error(typeck::invalid)]` or
        // `#[help(typeck::another_help)]`.
        let mut nested_iter = nested.into_iter();
        if let Some(nested_attr) = nested_iter.next() {
            // Report an error if there are any other list items after the path.
            if is_help_or_note && nested_iter.next().is_some() {
                throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                    diag.help("`help` and `note` struct attributes can only have one argument")
                });
            }

            match nested_attr {
                NestedMeta::Meta(Meta::Path(path)) if is_help_or_note => {
                    let fn_name = proc_macro2::Ident::new(name, attr.span());
                    return Ok(quote! { #diag.#fn_name(rustc_errors::fluent::#path); });
                }
                NestedMeta::Meta(Meta::Path(path)) => {
                    self.slug.set_once((path.clone(), span));
                }
                NestedMeta::Meta(meta @ Meta::NameValue(_))
                    if !is_help_or_note
                        && meta.path().segments.last().unwrap().ident.to_string() == "code" =>
                {
                    // don't error for valid follow-up attributes
                }
                nested_attr => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                    diag.help("first argument of the attribute should be the diagnostic slug")
                }),
            };
        }

        // Remaining attributes are optional, only `code = ".."` at the moment.
        let mut tokens = Vec::new();
        for nested_attr in nested_iter {
            let meta = match nested_attr {
                syn::NestedMeta::Meta(meta) => meta,
                _ => throw_invalid_nested_attr!(attr, &nested_attr),
            };

            let path = meta.path();
            let nested_name = path.segments.last().unwrap().ident.to_string();
            // Struct attributes are only allowed to be applied once, and the diagnostic
            // changes will be set in the initialisation code.
            if let Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(s), .. }) = &meta {
                let span = s.span().unwrap();
                match nested_name.as_str() {
                    "code" => {
                        self.code.set_once((s.value(), span));
                        let code = &self.code.as_ref().map(|(v, _)| v);
                        tokens.push(quote! {
                            #diag.code(rustc_errors::DiagnosticId::Error(#code.to_string()));
                        });
                    }
                    _ => invalid_nested_attr(attr, &nested_attr)
                        .help("only `code` is a valid nested attributes following the slug")
                        .emit(),
                }
            } else {
                invalid_nested_attr(attr, &nested_attr).emit()
            }
        }

        Ok(tokens.drain(..).collect())
    }

    fn generate_field_attrs_code(&mut self, binding_info: &BindingInfo<'_>) -> TokenStream {
        let field = binding_info.ast();
        let field_binding = &binding_info.binding;

        let inner_ty = FieldInnerTy::from_type(&field.ty);

        // When generating `set_arg` or `add_subdiagnostic` calls, move data rather than
        // borrow it to avoid requiring clones - this must therefore be the last use of
        // each field (for example, any formatting machinery that might refer to a field
        // should be generated already).
        if field.attrs.is_empty() {
            let diag = &self.diag;
            let ident = field.ident.as_ref().unwrap();
            quote! {
                #diag.set_arg(
                    stringify!(#ident),
                    #field_binding
                );
            }
        } else {
            field
                .attrs
                .iter()
                .map(move |attr| {
                    let name = attr.path.segments.last().unwrap().ident.to_string();
                    let (binding, needs_destructure) = match (name.as_str(), &inner_ty) {
                        // `primary_span` can accept a `Vec<Span>` so don't destructure that.
                        ("primary_span", FieldInnerTy::Vec(_)) => {
                            (quote! { #field_binding.clone() }, false)
                        }
                        // `subdiagnostics` are not derefed because they are bound by value.
                        ("subdiagnostic", _) => (quote! { #field_binding }, true),
                        _ => (quote! { *#field_binding }, true),
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
    }

    fn generate_inner_field_code(
        &mut self,
        attr: &Attribute,
        info: FieldInfo<'_>,
        binding: TokenStream,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let meta = attr.parse_meta()?;
        match meta {
            Meta::Path(_) => self.generate_inner_field_code_path(attr, info, binding),
            Meta::List(MetaList { .. }) => self.generate_inner_field_code_list(attr, info, binding),
            _ => throw_invalid_attr!(attr, &meta),
        }
    }

    fn generate_inner_field_code_path(
        &mut self,
        attr: &Attribute,
        info: FieldInfo<'_>,
        binding: TokenStream,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        assert!(matches!(attr.parse_meta()?, Meta::Path(_)));
        let diag = &self.diag;

        let meta = attr.parse_meta()?;

        let ident = &attr.path.segments.last().unwrap().ident;
        let name = ident.to_string();
        let name = name.as_str();
        match name {
            "skip_arg" => {
                // Don't need to do anything - by virtue of the attribute existing, the
                // `set_arg` call will not be generated.
                Ok(quote! {})
            }
            "primary_span" => {
                report_error_if_not_applied_to_span(attr, &info)?;
                Ok(quote! {
                    #diag.set_span(#binding);
                })
            }
            "label" => {
                report_error_if_not_applied_to_span(attr, &info)?;
                Ok(self.add_spanned_subdiagnostic(binding, ident, parse_quote! { _subdiag::label }))
            }
            "note" | "help" => {
                let path = match name {
                    "note" => parse_quote! { _subdiag::note },
                    "help" => parse_quote! { _subdiag::help },
                    _ => unreachable!(),
                };
                if type_matches_path(&info.ty, &["rustc_span", "Span"]) {
                    Ok(self.add_spanned_subdiagnostic(binding, ident, path))
                } else if type_is_unit(&info.ty) {
                    Ok(self.add_subdiagnostic(ident, path))
                } else {
                    report_type_error(attr, "`Span` or `()`")?;
                }
            }
            "subdiagnostic" => Ok(quote! { #diag.subdiagnostic(#binding); }),
            _ => throw_invalid_attr!(attr, &meta, |diag| {
                diag.help(
                    "only `skip_arg`, `primary_span`, `label`, `note`, `help` and `subdiagnostic` \
                     are valid field attributes",
                )
            }),
        }
    }

    fn generate_inner_field_code_list(
        &mut self,
        attr: &Attribute,
        info: FieldInfo<'_>,
        binding: TokenStream,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let meta = attr.parse_meta()?;
        let Meta::List(MetaList { ref path, ref nested, .. }) = meta  else { unreachable!() };

        let ident = &attr.path.segments.last().unwrap().ident;
        let name = path.segments.last().unwrap().ident.to_string();
        let name = name.as_ref();
        match name {
            "suggestion" | "suggestion_short" | "suggestion_hidden" | "suggestion_verbose" => {
                return self.generate_inner_field_code_suggestion(attr, info);
            }
            "label" | "help" | "note" => (),
            _ => throw_invalid_attr!(attr, &meta, |diag| {
                diag.help(
                    "only `label`, `note`, `help` or `suggestion{,_short,_hidden,_verbose}` are \
                     valid field attributes",
                )
            }),
        }

        // For `#[label(..)]`, `#[note(..)]` and `#[help(..)]`, the first nested element must be a
        // path, e.g. `#[label(typeck::label)]`.
        let mut nested_iter = nested.into_iter();
        let msg = match nested_iter.next() {
            Some(NestedMeta::Meta(Meta::Path(path))) => path.clone(),
            Some(nested_attr) => throw_invalid_nested_attr!(attr, &nested_attr),
            None => throw_invalid_attr!(attr, &meta),
        };

        // None of these attributes should have anything following the slug.
        if nested_iter.next().is_some() {
            throw_invalid_attr!(attr, &meta);
        }

        match name {
            "label" => {
                report_error_if_not_applied_to_span(attr, &info)?;
                Ok(self.add_spanned_subdiagnostic(binding, ident, msg))
            }
            "note" | "help" if type_matches_path(&info.ty, &["rustc_span", "Span"]) => {
                Ok(self.add_spanned_subdiagnostic(binding, ident, msg))
            }
            "note" | "help" if type_is_unit(&info.ty) => Ok(self.add_subdiagnostic(ident, msg)),
            "note" | "help" => {
                report_type_error(attr, "`Span` or `()`")?;
            }
            _ => unreachable!(),
        }
    }

    fn generate_inner_field_code_suggestion(
        &mut self,
        attr: &Attribute,
        info: FieldInfo<'_>,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let diag = &self.diag;

        let mut meta = attr.parse_meta()?;
        let Meta::List(MetaList { ref path, ref mut nested, .. }) = meta  else { unreachable!() };

        let (span_field, mut applicability) = self.span_and_applicability_of_ty(info)?;

        let mut msg = None;
        let mut code = None;

        let mut nested_iter = nested.into_iter().peekable();
        if let Some(nested_attr) = nested_iter.peek() {
            if let NestedMeta::Meta(Meta::Path(path)) = nested_attr {
                msg = Some(path.clone());
            }
        };
        // Move the iterator forward if a path was found (don't otherwise so that
        // code/applicability can be found or an error emitted).
        if msg.is_some() {
            let _ = nested_iter.next();
        }

        for nested_attr in nested_iter {
            let meta = match nested_attr {
                syn::NestedMeta::Meta(ref meta) => meta,
                syn::NestedMeta::Lit(_) => throw_invalid_nested_attr!(attr, &nested_attr),
            };

            let nested_name = meta.path().segments.last().unwrap().ident.to_string();
            let nested_name = nested_name.as_str();
            match meta {
                Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(s), .. }) => {
                    let span = meta.span().unwrap();
                    match nested_name {
                        "code" => {
                            let formatted_str = self.build_format(&s.value(), s.span());
                            code = Some(formatted_str);
                        }
                        "applicability" => {
                            applicability = match applicability {
                                Some(v) => {
                                    span_err(
                                        span,
                                        "applicability cannot be set in both the field and \
                                         attribute",
                                    )
                                    .emit();
                                    Some(v)
                                }
                                None => match Applicability::from_str(&s.value()) {
                                    Ok(v) => Some(quote! { #v }),
                                    Err(()) => {
                                        span_err(span, "invalid applicability").emit();
                                        None
                                    }
                                },
                            }
                        }
                        _ => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                            diag.help(
                                "only `message`, `code` and `applicability` are valid field \
                                 attributes",
                            )
                        }),
                    }
                }
                _ => throw_invalid_nested_attr!(attr, &nested_attr, |diag| {
                    if matches!(meta, Meta::Path(_)) {
                        diag.help("a diagnostic slug must be the first argument to the attribute")
                    } else {
                        diag
                    }
                }),
            }
        }

        let applicability =
            applicability.unwrap_or_else(|| quote!(rustc_errors::Applicability::Unspecified));

        let name = path.segments.last().unwrap().ident.to_string();
        let method = format_ident!("span_{}", name);

        let msg = msg.unwrap_or_else(|| parse_quote! { _subdiag::suggestion });
        let msg = quote! { rustc_errors::fluent::#msg };
        let code = code.unwrap_or_else(|| quote! { String::new() });

        Ok(quote! { #diag.#method(#span_field, #msg, #code, #applicability); })
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
    ) -> Result<(TokenStream, Option<TokenStream>), SessionDiagnosticDeriveError> {
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

                for (idx, elem) in tup.elems.iter().enumerate() {
                    if type_matches_path(elem, &["rustc_span", "Span"]) {
                        if span_idx.is_none() {
                            span_idx = Some(syn::Index::from(idx));
                        } else {
                            throw_span_err!(
                                info.span.unwrap(),
                                "type of field annotated with `#[suggestion(...)]` contains more \
                                 than one `Span`"
                            );
                        }
                    } else if type_matches_path(elem, &["rustc_errors", "Applicability"]) {
                        if applicability_idx.is_none() {
                            applicability_idx = Some(syn::Index::from(idx));
                        } else {
                            throw_span_err!(
                                info.span.unwrap(),
                                "type of field annotated with `#[suggestion(...)]` contains more \
                                 than one Applicability"
                            );
                        }
                    }
                }

                if let Some(span_idx) = span_idx {
                    let binding = &info.binding.binding;
                    let span = quote!(#binding.#span_idx);
                    let applicability = applicability_idx
                        .map(|applicability_idx| quote!(#binding.#applicability_idx))
                        .unwrap_or_else(|| quote!(rustc_errors::Applicability::Unspecified));

                    return Ok((span, Some(applicability)));
                }

                throw_span_err!(info.span.unwrap(), "wrong types for suggestion", |diag| {
                    diag.help(
                        "`#[suggestion(...)]` on a tuple field must be applied to fields of type \
                         `(Span, Applicability)`",
                    )
                });
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
