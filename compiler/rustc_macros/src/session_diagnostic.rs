#![deny(unused_must_use)]
use proc_macro::{Diagnostic, Level, MultiSpan};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::str::FromStr;
use syn::{spanned::Spanned, Attribute, Meta, MetaList, MetaNameValue, Type, Visibility};
use synstructure::{BindingInfo, Structure, VariantInfo};

/// Implements `#[derive(SessionDiagnostic)]`, which allows for errors to be specified as a struct,
/// independent from the actual diagnostics emitting code.
///
/// ```ignore (pseudo-rust)
/// # extern crate rustc_errors;
/// # use rustc_errors::Applicability;
/// # extern crate rustc_span;
/// # use rustc_span::{symbol::Ident, Span};
/// # extern crate rust_middle;
/// # use rustc_middle::ty::Ty;
/// #[derive(SessionDiagnostic)]
/// #[error(code = "E0505", slug = "borrowck-move-out-of-borrow")]
/// pub struct MoveOutOfBorrowError<'tcx> {
///     pub name: Ident,
///     pub ty: Ty<'tcx>,
///     #[primary_span]
///     #[label]
///     pub span: Span,
///     #[label = "first-borrow-label"]
///     pub first_borrow_span: Span,
///     #[suggestion(code = "{name}.clone()")]
///     pub clone_sugg: Option<(Span, Applicability)>
/// }
/// ```
///
/// ```fluent
/// move-out-of-borrow = cannot move out of {$name} because it is borrowed
///     .label = cannot move out of borrow
///     .first-borrow-label = `{$ty}` first borrowed here
///     .suggestion = consider cloning here
/// ```
///
/// Then, later, to emit the error:
///
/// ```ignore (pseudo-rust)
/// sess.emit_err(MoveOutOfBorrowError {
///     expected,
///     actual,
///     span,
///     first_borrow_span,
///     clone_sugg: Some(suggestion, Applicability::MachineApplicable),
/// });
/// ```
///
/// See rustc dev guide for more examples on using the `#[derive(SessionDiagnostic)]`:
/// <https://rustc-dev-guide.rust-lang.org/diagnostics/sessiondiagnostic.html>
pub fn session_diagnostic_derive(s: Structure<'_>) -> TokenStream {
    // Names for the diagnostic we build and the session we build it from.
    let diag = format_ident!("diag");
    let sess = format_ident!("sess");

    SessionDiagnosticDerive::new(diag, sess, s).into_tokens()
}

/// Implements `#[derive(SessionSubdiagnostic)]`, which allows for labels, notes, helps and
/// suggestions to be specified as a structs or enums, independent from the actual diagnostics
/// emitting code or diagnostic derives.
///
/// ```ignore (pseudo-rust)
/// #[derive(SessionSubdiagnostic)]
/// pub enum ExpectedIdentifierLabel<'tcx> {
///     #[label(slug = "parser-expected-identifier")]
///     WithoutFound {
///         #[primary_span]
///         span: Span,
///     }
///     #[label(slug = "parser-expected-identifier-found")]
///     WithFound {
///         #[primary_span]
///         span: Span,
///         found: String,
///     }
/// }
///
/// #[derive(SessionSubdiagnostic)]
/// #[suggestion_verbose(slug = "parser-raw-identifier")]
/// pub struct RawIdentifierSuggestion<'tcx> {
///     #[primary_span]
///     span: Span,
///     #[applicability]
///     applicability: Applicability,
///     ident: Ident,
/// }
/// ```
///
/// ```fluent
/// parser-expected-identifier = expected identifier
///
/// parser-expected-identifier-found = expected identifier, found {$found}
///
/// parser-raw-identifier = escape `{$ident}` to use it as an identifier
/// ```
///
/// Then, later, to add the subdiagnostic:
///
/// ```ignore (pseudo-rust)
/// diag.subdiagnostic(ExpectedIdentifierLabel::WithoutFound { span });
///
/// diag.subdiagnostic(RawIdentifierSuggestion { span, applicability, ident });
/// ```
pub fn session_subdiagnostic_derive(s: Structure<'_>) -> TokenStream {
    SessionSubdiagnosticDerive::new(s).into_tokens()
}

/// Checks whether the type name of `ty` matches `name`.
///
/// Given some struct at `a::b::c::Foo`, this will return true for `c::Foo`, `b::c::Foo`, or
/// `a::b::c::Foo`. This reasonably allows qualified names to be used in the macro.
fn type_matches_path(ty: &Type, name: &[&str]) -> bool {
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

/// The central struct for constructing the `as_error` method from an annotated struct.
struct SessionDiagnosticDerive<'a> {
    structure: Structure<'a>,
    builder: SessionDiagnosticDeriveBuilder,
}

impl std::convert::From<syn::Error> for SessionDiagnosticDeriveError {
    fn from(e: syn::Error) -> Self {
        SessionDiagnosticDeriveError::SynError(e)
    }
}

#[derive(Debug)]
enum SessionDiagnosticDeriveError {
    SynError(syn::Error),
    ErrorHandled,
}

impl SessionDiagnosticDeriveError {
    fn to_compile_error(self) -> TokenStream {
        match self {
            SessionDiagnosticDeriveError::SynError(e) => e.to_compile_error(),
            SessionDiagnosticDeriveError::ErrorHandled => {
                // Return ! to avoid having to create a blank DiagnosticBuilder to return when an
                // error has already been emitted to the compiler.
                quote! {
                    { unreachable!(); }
                }
            }
        }
    }
}

fn span_err(span: impl MultiSpan, msg: &str) -> Diagnostic {
    Diagnostic::spanned(span, Level::Error, msg)
}

/// For methods that return a `Result<_, SessionDiagnosticDeriveError>`:
///
/// Emit a diagnostic on span `$span` with msg `$msg` (optionally performing additional decoration
/// using the `FnOnce` passed in `diag`) and return `Err(ErrorHandled)`.
macro_rules! throw_span_err {
    ($span:expr, $msg:expr) => {{ throw_span_err!($span, $msg, |diag| diag) }};
    ($span:expr, $msg:expr, $f:expr) => {{
        return Err(_throw_span_err($span, $msg, $f));
    }};
}

/// When possible, prefer using `throw_span_err!` over using this function directly. This only
/// exists as a function to constrain `f` to an `impl FnOnce`.
fn _throw_span_err(
    span: impl MultiSpan,
    msg: &str,
    f: impl FnOnce(Diagnostic) -> Diagnostic,
) -> SessionDiagnosticDeriveError {
    let diag = span_err(span, msg);
    f(diag).emit();
    SessionDiagnosticDeriveError::ErrorHandled
}

impl<'a> SessionDiagnosticDerive<'a> {
    fn new(diag: syn::Ident, sess: syn::Ident, structure: Structure<'a>) -> Self {
        // Build the mapping of field names to fields. This allows attributes to peek values from
        // other fields.
        let mut fields_map = HashMap::new();

        // Convenience bindings.
        let ast = structure.ast();

        if let syn::Data::Struct(syn::DataStruct { fields, .. }) = &ast.data {
            for field in fields.iter() {
                if let Some(ident) = &field.ident {
                    fields_map.insert(ident.to_string(), quote! { &self.#ident });
                }
            }
        }

        Self {
            builder: SessionDiagnosticDeriveBuilder {
                diag,
                sess,
                fields: fields_map,
                kind: None,
                code: None,
                slug: None,
            },
            structure,
        }
    }

    fn into_tokens(self) -> TokenStream {
        let SessionDiagnosticDerive { mut structure, mut builder } = self;

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

                // Generates calls to `span_label` and similar functions based on the attributes
                // on fields. Code for suggestions uses formatting machinery and the value of
                // other fields - because any given field can be referenced multiple times, it
                // should be accessed through a borrow. When passing fields to `set_arg` (which
                // happens below) for Fluent, we want to move the data, so that has to happen
                // in a separate pass over the fields.
                let attrs = structure.each(|field_binding| {
                    let field = field_binding.ast();
                    let result = field.attrs.iter().map(|attr| {
                        builder
                            .generate_field_attr_code(
                                attr,
                                FieldInfo {
                                    vis: &field.vis,
                                    binding: field_binding,
                                    ty: &field.ty,
                                    span: &field.span(),
                                },
                            )
                            .unwrap_or_else(|v| v.to_compile_error())
                    });

                    quote! { #(#result);* }
                });

                // When generating `set_arg` calls, move data rather than borrow it to avoid
                // requiring clones - this must therefore be the last use of each field (for
                // example, any formatting machinery that might refer to a field should be
                // generated already).
                structure.bind_with(|_| synstructure::BindStyle::Move);
                let args = structure.each(|field_binding| {
                    let field = field_binding.ast();
                    // When a field has attributes like `#[label]` or `#[note]` then it doesn't
                    // need to be passed as an argument to the diagnostic. But when a field has no
                    // attributes then it must be passed as an argument to the diagnostic so that
                    // it can be referred to by Fluent messages.
                    if field.attrs.is_empty() {
                        let diag = &builder.diag;
                        let ident = field_binding.ast().ident.as_ref().unwrap();
                        quote! {
                            #diag.set_arg(
                                stringify!(#ident),
                                #field_binding.into_diagnostic_arg()
                            );
                        }
                    } else {
                        quote! {}
                    }
                });

                let span = ast.span().unwrap();
                let (diag, sess) = (&builder.diag, &builder.sess);
                let init = match (builder.kind, builder.slug) {
                    (None, _) => {
                        span_err(span, "diagnostic kind not specified")
                            .help("use the `#[error(...)]` attribute to create an error")
                            .emit();
                        return SessionDiagnosticDeriveError::ErrorHandled.to_compile_error();
                    }
                    (Some((kind, _)), None) => {
                        span_err(span, "`slug` not specified")
                            .help(&format!("use the `#[{}(slug = \"...\")]` attribute to set this diagnostic's slug", kind.descr()))
                            .emit();
                        return SessionDiagnosticDeriveError::ErrorHandled.to_compile_error();
                    }
                    (Some((SessionDiagnosticKind::Error, _)), Some((slug, _))) => {
                        quote! {
                            let mut #diag = #sess.struct_err(
                                rustc_errors::DiagnosticMessage::fluent(#slug),
                            );
                        }
                    }
                    (Some((SessionDiagnosticKind::Warn, _)), Some((slug, _))) => {
                        quote! {
                            let mut #diag = #sess.struct_warn(
                                rustc_errors::DiagnosticMessage::fluent(#slug),
                            );
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

        let sess = &builder.sess;
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

/// Field information passed to the builder. Deliberately omits attrs to discourage the
/// `generate_*` methods from walking the attributes themselves.
struct FieldInfo<'a> {
    vis: &'a Visibility,
    binding: &'a BindingInfo<'a>,
    ty: &'a Type,
    span: &'a proc_macro2::Span,
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
    /// Name of the session parameter that's passed in to the `as_error` method.
    sess: syn::Ident,
    /// The identifier to use for the generated `DiagnosticBuilder` instance.
    diag: syn::Ident,

    /// Store a map of field name to its corresponding field. This is built on construction of the
    /// derive builder.
    fields: HashMap<String, TokenStream>,

    /// Kind of diagnostic requested via the struct attribute.
    kind: Option<(SessionDiagnosticKind, proc_macro::Span)>,
    /// Slug is a mandatory part of the struct attribute as corresponds to the Fluent message that
    /// has the actual diagnostic message.
    slug: Option<(String, proc_macro::Span)>,
    /// Error codes are a optional part of the struct attribute - this is only set to detect
    /// multiple specifications.
    code: Option<proc_macro::Span>,
}

impl SessionDiagnosticDeriveBuilder {
    /// Establishes state in the `SessionDiagnosticDeriveBuilder` resulting from the struct
    /// attributes like `#[error(..)#`, such as the diagnostic kind and slug. Generates
    /// diagnostic builder calls for setting error code and creating note/help messages.
    fn generate_structure_code(
        &mut self,
        attr: &Attribute,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let span = attr.span().unwrap();

        let name = attr.path.segments.last().unwrap().ident.to_string();
        let name = name.as_str();
        let meta = attr.parse_meta()?;

        if matches!(name, "help" | "note") && matches!(meta, Meta::Path(_) | Meta::NameValue(_)) {
            let diag = &self.diag;
            let slug = match &self.slug {
                Some((slug, _)) => slug.as_str(),
                None => throw_span_err!(
                    span,
                    &format!(
                        "`#[{}{}]` must come after `#[error(..)]` or `#[warn(..)]`",
                        name,
                        match meta {
                            Meta::Path(_) => "",
                            Meta::NameValue(_) => " = ...",
                            _ => unreachable!(),
                        }
                    )
                ),
            };
            let id = match meta {
                Meta::Path(..) => quote! { #name },
                Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(s), .. }) => {
                    quote! { #s }
                }
                _ => unreachable!(),
            };
            let fn_name = proc_macro2::Ident::new(name, attr.span());

            return Ok(quote! {
                #diag.#fn_name(rustc_errors::DiagnosticMessage::fluent_attr(#slug, #id));
            });
        }

        let nested = match meta {
            Meta::List(MetaList { nested, .. }) => nested,
            Meta::Path(..) => throw_span_err!(
                span,
                &format!("`#[{}]` is not a valid `SessionDiagnostic` struct attribute", name)
            ),
            Meta::NameValue(..) => throw_span_err!(
                span,
                &format!("`#[{} = ...]` is not a valid `SessionDiagnostic` struct attribute", name)
            ),
        };

        let kind = match name {
            "error" => SessionDiagnosticKind::Error,
            "warning" => SessionDiagnosticKind::Warn,
            other => throw_span_err!(
                span,
                &format!("`#[{}(...)]` is not a valid `SessionDiagnostic` struct attribute", other)
            ),
        };
        self.set_kind_once(kind, span)?;

        let mut tokens = Vec::new();
        for attr in nested {
            let span = attr.span().unwrap();
            let meta = match attr {
                syn::NestedMeta::Meta(meta) => meta,
                syn::NestedMeta::Lit(_) => throw_span_err!(
                    span,
                    &format!(
                        "`#[{}(\"...\")]` is not a valid `SessionDiagnostic` struct attribute",
                        name
                    )
                ),
            };

            let path = meta.path();
            let nested_name = path.segments.last().unwrap().ident.to_string();
            match &meta {
                // Struct attributes are only allowed to be applied once, and the diagnostic
                // changes will be set in the initialisation code.
                Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(s), .. }) => {
                    match nested_name.as_str() {
                        "slug" => {
                            self.set_slug_once(s.value(), s.span().unwrap());
                        }
                        "code" => {
                            tokens.push(self.set_code_once(s.value(), s.span().unwrap()));
                        }
                        other => {
                            let diag = span_err(
                                span,
                                &format!(
                                    "`#[{}({} = ...)]` is not a valid `SessionDiagnostic` struct attribute",
                                    name, other
                                ),
                            );
                            diag.emit();
                        }
                    }
                }
                Meta::NameValue(..) => {
                    span_err(
                        span,
                        &format!(
                            "`#[{}({} = ...)]` is not a valid `SessionDiagnostic` struct attribute",
                            name, nested_name
                        ),
                    )
                    .help("value must be a string")
                    .emit();
                }
                Meta::Path(..) => {
                    span_err(
                        span,
                        &format!(
                            "`#[{}({})]` is not a valid `SessionDiagnostic` struct attribute",
                            name, nested_name
                        ),
                    )
                    .emit();
                }
                Meta::List(..) => {
                    span_err(
                        span,
                        &format!(
                            "`#[{}({}(...))]` is not a valid `SessionDiagnostic` struct attribute",
                            name, nested_name
                        ),
                    )
                    .emit();
                }
            }
        }

        Ok(tokens.drain(..).collect())
    }

    #[must_use]
    fn set_kind_once(
        &mut self,
        kind: SessionDiagnosticKind,
        span: proc_macro::Span,
    ) -> Result<(), SessionDiagnosticDeriveError> {
        match self.kind {
            None => {
                self.kind = Some((kind, span));
                Ok(())
            }
            Some((prev_kind, prev_span)) => {
                let existing = prev_kind.descr();
                let current = kind.descr();

                let msg = if current == existing {
                    format!("`{}` specified multiple times", existing)
                } else {
                    format!("`{}` specified when `{}` was already specified", current, existing)
                };
                throw_span_err!(span, &msg, |diag| diag
                    .span_note(prev_span, "previously specified here"));
            }
        }
    }

    fn set_code_once(&mut self, code: String, span: proc_macro::Span) -> TokenStream {
        match self.code {
            None => {
                self.code = Some(span);
            }
            Some(prev_span) => {
                span_err(span, "`code` specified multiple times")
                    .span_note(prev_span, "previously specified here")
                    .emit();
            }
        }

        let diag = &self.diag;
        quote! { #diag.code(rustc_errors::DiagnosticId::Error(#code.to_string())); }
    }

    fn set_slug_once(&mut self, slug: String, span: proc_macro::Span) {
        match self.slug {
            None => {
                self.slug = Some((slug, span));
            }
            Some((_, prev_span)) => {
                span_err(span, "`slug` specified multiple times")
                    .span_note(prev_span, "previously specified here")
                    .emit();
            }
        }
    }

    fn generate_field_attr_code(
        &mut self,
        attr: &syn::Attribute,
        info: FieldInfo<'_>,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let field_binding = &info.binding.binding;
        let option_ty = option_inner_ty(&info.ty);
        let generated_code = self.generate_non_option_field_code(
            attr,
            FieldInfo {
                vis: info.vis,
                binding: info.binding,
                ty: option_ty.unwrap_or(&info.ty),
                span: info.span,
            },
        )?;

        if option_ty.is_none() {
            Ok(quote! { #generated_code })
        } else {
            Ok(quote! {
                if let Some(#field_binding) = #field_binding {
                    #generated_code
                }
            })
        }
    }

    fn generate_non_option_field_code(
        &mut self,
        attr: &Attribute,
        info: FieldInfo<'_>,
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let diag = &self.diag;
        let span = attr.span().unwrap();
        let field_binding = &info.binding.binding;

        let name = attr.path.segments.last().unwrap().ident.to_string();
        let name = name.as_str();

        let meta = attr.parse_meta()?;
        match meta {
            Meta::Path(_) => match name {
                "skip_arg" => {
                    // Don't need to do anything - by virtue of the attribute existing, the
                    // `set_arg` call will not be generated.
                    Ok(quote! {})
                }
                "primary_span" => {
                    report_error_if_not_applied_to_span(attr, &info)?;
                    Ok(quote! {
                        #diag.set_span(*#field_binding);
                    })
                }
                "label" | "note" | "help" => {
                    report_error_if_not_applied_to_span(attr, &info)?;
                    Ok(self.add_subdiagnostic(field_binding, name, name))
                }
                other => throw_span_err!(
                    span,
                    &format!("`#[{}]` is not a valid `SessionDiagnostic` field attribute", other)
                ),
            },
            Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(s), .. }) => match name {
                "label" | "note" | "help" => {
                    report_error_if_not_applied_to_span(attr, &info)?;
                    Ok(self.add_subdiagnostic(field_binding, name, &s.value()))
                }
                other => throw_span_err!(
                    span,
                    &format!(
                        "`#[{} = ...]` is not a valid `SessionDiagnostic` field attribute",
                        other
                    )
                ),
            },
            Meta::NameValue(_) => throw_span_err!(
                span,
                &format!("`#[{} = ...]` is not a valid `SessionDiagnostic` field attribute", name),
                |diag| diag.help("value must be a string")
            ),
            Meta::List(MetaList { path, nested, .. }) => {
                let name = path.segments.last().unwrap().ident.to_string();
                let name = name.as_ref();

                match name {
                    "suggestion" | "suggestion_short" | "suggestion_hidden"
                    | "suggestion_verbose" => (),
                    other => throw_span_err!(
                        span,
                        &format!(
                            "`#[{}(...)]` is not a valid `SessionDiagnostic` field attribute",
                            other
                        )
                    ),
                };

                let (span_, applicability) = self.span_and_applicability_of_ty(info)?;

                let mut msg = None;
                let mut code = None;

                for attr in nested {
                    let meta = match attr {
                        syn::NestedMeta::Meta(meta) => meta,
                        syn::NestedMeta::Lit(_) => throw_span_err!(
                            span,
                            &format!(
                                "`#[{}(\"...\")]` is not a valid `SessionDiagnostic` field attribute",
                                name
                            )
                        ),
                    };

                    let span = meta.span().unwrap();
                    let nested_name = meta.path().segments.last().unwrap().ident.to_string();
                    let nested_name = nested_name.as_str();

                    match meta {
                        Meta::NameValue(MetaNameValue { lit: syn::Lit::Str(s), .. }) => {
                            match nested_name {
                                "message" => {
                                    msg = Some(s.value());
                                }
                                "code" => {
                                    let formatted_str = self.build_format(&s.value(), s.span());
                                    code = Some(formatted_str);
                                }
                                other => throw_span_err!(
                                    span,
                                    &format!(
                                        "`#[{}({} = ...)]` is not a valid `SessionDiagnostic` field attribute",
                                        name, other
                                    )
                                ),
                            }
                        }
                        Meta::NameValue(..) => throw_span_err!(
                            span,
                            &format!(
                                "`#[{}({} = ...)]` is not a valid `SessionDiagnostic` struct attribute",
                                name, nested_name
                            ),
                            |diag| diag.help("value must be a string")
                        ),
                        Meta::Path(..) => throw_span_err!(
                            span,
                            &format!(
                                "`#[{}({})]` is not a valid `SessionDiagnostic` struct attribute",
                                name, nested_name
                            )
                        ),
                        Meta::List(..) => throw_span_err!(
                            span,
                            &format!(
                                "`#[{}({}(...))]` is not a valid `SessionDiagnostic` struct attribute",
                                name, nested_name
                            )
                        ),
                    }
                }

                let method = format_ident!("span_{}", name);

                let slug = self
                    .slug
                    .as_ref()
                    .map(|(slug, _)| slug.as_str())
                    .unwrap_or_else(|| "missing-slug");
                let msg = msg.as_deref().unwrap_or("suggestion");
                let msg = quote! { rustc_errors::DiagnosticMessage::fluent_attr(#slug, #msg) };
                let code = code.unwrap_or_else(|| quote! { String::new() });

                Ok(quote! { #diag.#method(#span_, #msg, #code, #applicability); })
            }
        }
    }

    /// Adds a subdiagnostic by generating a `diag.span_$kind` call with the current slug and
    /// `fluent_attr_identifier`.
    fn add_subdiagnostic(
        &self,
        field_binding: &proc_macro2::Ident,
        kind: &str,
        fluent_attr_identifier: &str,
    ) -> TokenStream {
        let diag = &self.diag;

        let slug =
            self.slug.as_ref().map(|(slug, _)| slug.as_str()).unwrap_or_else(|| "missing-slug");
        let fn_name = format_ident!("span_{}", kind);
        quote! {
            #diag.#fn_name(
                *#field_binding,
                rustc_errors::DiagnosticMessage::fluent_attr(#slug, #fluent_attr_identifier)
            );
        }
    }

    fn span_and_applicability_of_ty(
        &self,
        info: FieldInfo<'_>,
    ) -> Result<(TokenStream, TokenStream), SessionDiagnosticDeriveError> {
        match &info.ty {
            // If `ty` is `Span` w/out applicability, then use `Applicability::Unspecified`.
            ty @ Type::Path(..) if type_matches_path(ty, &["rustc_span", "Span"]) => {
                let binding = &info.binding.binding;
                Ok((quote!(*#binding), quote!(rustc_errors::Applicability::Unspecified)))
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
                                "type of field annotated with `#[suggestion(...)]` contains more than one `Span`"
                            );
                        }
                    } else if type_matches_path(elem, &["rustc_errors", "Applicability"]) {
                        if applicability_idx.is_none() {
                            applicability_idx = Some(syn::Index::from(idx));
                        } else {
                            throw_span_err!(
                                info.span.unwrap(),
                                "type of field annotated with `#[suggestion(...)]` contains more than one Applicability"
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

                    return Ok((span, applicability));
                }

                throw_span_err!(info.span.unwrap(), "wrong types for suggestion", |diag| {
                    diag.help("`#[suggestion(...)]` on a tuple field must be applied to fields of type `(Span, Applicability)`")
                });
            }
            // If `ty` isn't a `Span` or `(Span, Applicability)` then emit an error.
            _ => throw_span_err!(info.span.unwrap(), "wrong field type for suggestion", |diag| {
                diag.help("`#[suggestion(...)]` should be applied to fields of type `Span` or `(Span, Applicability)`")
            }),
        }
    }
}

trait HasFieldMap {
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
                    Diagnostic::spanned(
                        span.unwrap(),
                        Level::Error,
                        format!("`{}` doesn't refer to a field on this type", field),
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

impl HasFieldMap for SessionDiagnosticDeriveBuilder {
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream> {
        self.fields.get(field)
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
fn report_error_if_not_applied_to_applicability(
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
fn report_error_if_not_applied_to_span(
    attr: &Attribute,
    info: &FieldInfo<'_>,
) -> Result<(), SessionDiagnosticDeriveError> {
    report_error_if_not_applied_to_ty(attr, info, &["rustc_span", "Span"], "Span")
}

/// If `ty` is an Option, returns `Some(inner type)`, otherwise returns `None`.
fn option_inner_ty(ty: &Type) -> Option<&Type> {
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

trait SetOnce<T> {
    fn set_once(&mut self, value: T);
}

impl<T> SetOnce<(T, proc_macro::Span)> for Option<(T, proc_macro::Span)> {
    fn set_once(&mut self, (value, span): (T, proc_macro::Span)) {
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

enum Applicability {
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

#[derive(Clone, Copy)]
enum SubdiagnosticKind {
    /// `#[label]` or `#[label(...)]`
    Label,
    /// `#[note]` or  `#[note(...)]`
    Note,
    /// `#[help]` or `#[help(...)]`
    Help,
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

struct SessionSubdiagnosticDerive<'a> {
    structure: Structure<'a>,
    diag: syn::Ident,
}

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
    /// `#[kind(slug = "...")]` attribute on the type or variant.
    slug: Option<(String, proc_macro::Span)>,
    /// If a suggestion, the code to suggest as a replacement - from the `#[kind(code = "...")]`
    /// attribute on the type or variant.
    code: Option<(proc_macro2::TokenStream, proc_macro::Span)>,

    /// Identifier for the binding to the `#[primary_span]` field.
    span_field: Option<(proc_macro2::Ident, proc_macro::Span)>,
    /// If a suggestion, the identifier for the binding to the `#[applicability]` field or a
    /// `rustc_errors::Applicability::*` variant directly.
    applicability: Option<(proc_macro2::TokenStream, proc_macro::Span)>,
}

impl<'a> HasFieldMap for SessionSubdiagnosticDeriveBuilder<'a> {
    fn get_field_binding(&self, field: &String) -> Option<&TokenStream> {
        self.fields.get(field)
    }
}

impl<'a> SessionSubdiagnosticDeriveBuilder<'a> {
    fn identify_kind(&mut self) -> Result<(), SessionDiagnosticDeriveError> {
        for attr in self.variant.ast().attrs {
            let span = attr.span().unwrap();

            let name = attr.path.segments.last().unwrap().ident.to_string();
            let name = name.as_str();

            let meta = attr.parse_meta()?;
            let kind = match meta {
                Meta::Path(_) => throw_span_err!(
                    span,
                    &format!("`#[{}]` is not a valid `SessionSubdiagnostic` attribute", name)
                ),
                Meta::NameValue(_) => throw_span_err!(
                    span,
                    &format!("`#[{} = ...]` is not a valid `SessionSubdiagnostic` attribute", name)
                ),
                Meta::List(MetaList { nested, .. }) => {
                    for attr in nested {
                        let meta = match attr {
                            syn::NestedMeta::Meta(meta) => meta,
                            syn::NestedMeta::Lit(_) => throw_span_err!(
                                span,
                                &format!(
                                    "`#[{}(\"...\")]` is not a valid `SessionSubdiagnostic` attribute",
                                    name
                                )
                            ),
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
                                    "slug" => self.slug.set_once((s.value(), span)),
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
                                    other => throw_span_err!(
                                        span,
                                        &format!(
                                            "`#[{}({} = ...)]` is not a valid `SessionSubdiagnostic` attribute",
                                            name, other
                                        )
                                    ),
                                }
                            }
                            Meta::NameValue(..) => throw_span_err!(
                                span,
                                &format!(
                                    "`#[{}({} = ...)]` is not a valid `SessionSubdiagnostic` attribute",
                                    name, nested_name
                                ),
                                |diag| diag.help("value must be a string")
                            ),
                            Meta::Path(..) => throw_span_err!(
                                span,
                                &format!(
                                    "`#[{}({})]` is not a valid `SessionSubdiagnostic` attribute",
                                    name, nested_name
                                )
                            ),
                            Meta::List(..) => throw_span_err!(
                                span,
                                &format!(
                                    "`#[{}({}(...))]` is not a valid `SessionSubdiagnostic` attribute",
                                    name, nested_name
                                )
                            ),
                        }
                    }

                    let Ok(kind) = SubdiagnosticKind::from_str(name) else {
                        throw_span_err!(
                            span,
                            &format!(
                                "`#[{}(...)]` is not a valid `SessionSubdiagnostic` attribute",
                                name
                            )
                        );
                    };
                    kind
                }
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

            if self.slug.is_none() {
                throw_span_err!(
                    span,
                    &format!("`slug` must be set in a `#[{}(...)]` attribute", name)
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
    ) -> Result<TokenStream, SessionDiagnosticDeriveError> {
        let ast = binding.ast();

        let option_ty = option_inner_ty(&ast.ty);
        let info = FieldInfo {
            vis: &ast.vis,
            binding: binding,
            ty: option_ty.unwrap_or(&ast.ty),
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
                    other => span_err(
                        span,
                        &format!(
                            "`#[{}]` is not a valid `SessionSubdiagnostic` field attribute",
                            other
                        ),
                    )
                    .emit(),
                },
                Meta::NameValue(_) => span_err(
                    span,
                    &format!(
                        "`#[{} = ...]` is not a valid `SessionSubdiagnostic` field attribute",
                        name
                    ),
                )
                .emit(),
                Meta::List(_) => span_err(
                    span,
                    &format!(
                        "`#[{}(...)]` is not a valid `SessionSubdiagnostic` field attribute",
                        name
                    ),
                )
                .emit(),
            }
        }

        let ident = ast.ident.as_ref().unwrap();

        let diag = &self.diag;
        let generated = quote! {
            #diag.set_arg(
                stringify!(#ident),
                #binding.into_diagnostic_arg()
            );
        };

        if option_ty.is_none() {
            Ok(quote! { #generated })
        } else {
            Ok(quote! {
                if let Some(#binding) = #binding {
                    #generated
                }
            })
        }
    }

    fn into_tokens(&mut self) -> Result<TokenStream, SessionDiagnosticDeriveError> {
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
        let slug = self.slug.as_ref().map(|(slug, _)| &**slug).unwrap_or("missing-slug");
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
        let message = quote! { rustc_errors::DiagnosticMessage::fluent(#slug) };
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

impl<'a> SessionSubdiagnosticDerive<'a> {
    fn new(structure: Structure<'a>) -> Self {
        let diag = format_ident!("diag");
        Self { structure, diag }
    }

    fn into_tokens(self) -> TokenStream {
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
