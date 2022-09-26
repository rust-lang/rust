#![deny(unused_must_use)]

use crate::diagnostics::diagnostic_builder::{DiagnosticDeriveBuilder, DiagnosticDeriveKind};
use crate::diagnostics::error::{span_err, DiagnosticDeriveError};
use crate::diagnostics::utils::{build_field_mapping, SetOnce};
use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use synstructure::Structure;

/// The central struct for constructing the `into_diagnostic` method from an annotated struct.
pub(crate) struct DiagnosticDerive<'a> {
    structure: Structure<'a>,
    handler: syn::Ident,
    builder: DiagnosticDeriveBuilder,
}

impl<'a> DiagnosticDerive<'a> {
    pub(crate) fn new(diag: syn::Ident, handler: syn::Ident, structure: Structure<'a>) -> Self {
        Self {
            builder: DiagnosticDeriveBuilder {
                diag,
                fields: build_field_mapping(&structure),
                kind: DiagnosticDeriveKind::Diagnostic,
                code: None,
                slug: None,
            },
            handler,
            structure,
        }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let DiagnosticDerive { mut structure, handler, mut builder } = self;

        let ast = structure.ast();
        let implementation = {
            if let syn::Data::Struct(..) = ast.data {
                let preamble = builder.preamble(&structure);
                let (attrs, args) = builder.body(&mut structure);

                let span = ast.span().unwrap();
                let diag = &builder.diag;
                let init = match builder.slug.value() {
                    None => {
                        span_err(span, "diagnostic slug not specified")
                            .help(&format!(
                                "specify the slug as the first argument to the `#[diag(...)]` attribute, \
                                such as `#[diag(typeck::example_error)]`",
                            ))
                            .emit();
                        return DiagnosticDeriveError::ErrorHandled.to_compile_error();
                    }
                    Some(slug) => {
                        quote! {
                            let mut #diag = #handler.struct_diagnostic(rustc_errors::fluent::#slug);
                        }
                    }
                };

                quote! {
                    #init
                    #preamble
                    match self {
                        #attrs
                    }
                    match self {
                        #args
                    }
                    #diag
                }
            } else {
                span_err(
                    ast.span().unwrap(),
                    "`#[derive(Diagnostic)]` can only be used on structs",
                )
                .emit();

                DiagnosticDeriveError::ErrorHandled.to_compile_error()
            }
        };

        structure.gen_impl(quote! {
            gen impl<'__diagnostic_handler_sess, G>
                    rustc_errors::IntoDiagnostic<'__diagnostic_handler_sess, G>
                    for @Self
                where G: rustc_errors::EmissionGuarantee
            {
                fn into_diagnostic(
                    self,
                    #handler: &'__diagnostic_handler_sess rustc_errors::Handler
                ) -> rustc_errors::DiagnosticBuilder<'__diagnostic_handler_sess, G> {
                    use rustc_errors::IntoDiagnosticArg;
                    #implementation
                }
            }
        })
    }
}

/// The central struct for constructing the `decorate_lint` method from an annotated struct.
pub(crate) struct LintDiagnosticDerive<'a> {
    structure: Structure<'a>,
    builder: DiagnosticDeriveBuilder,
}

impl<'a> LintDiagnosticDerive<'a> {
    pub(crate) fn new(diag: syn::Ident, structure: Structure<'a>) -> Self {
        Self {
            builder: DiagnosticDeriveBuilder {
                diag,
                fields: build_field_mapping(&structure),
                kind: DiagnosticDeriveKind::LintDiagnostic,
                code: None,
                slug: None,
            },
            structure,
        }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let LintDiagnosticDerive { mut structure, mut builder } = self;

        let ast = structure.ast();
        let implementation = {
            if let syn::Data::Struct(..) = ast.data {
                let preamble = builder.preamble(&structure);
                let (attrs, args) = builder.body(&mut structure);

                let diag = &builder.diag;
                let span = ast.span().unwrap();
                let init = match builder.slug.value() {
                    None => {
                        span_err(span, "diagnostic slug not specified")
                            .help(&format!(
                                "specify the slug as the first argument to the attribute, such as \
                                 `#[diag(typeck::example_error)]`",
                            ))
                            .emit();
                        return DiagnosticDeriveError::ErrorHandled.to_compile_error();
                    }
                    Some(slug) => {
                        quote! {
                            let mut #diag = #diag.build(rustc_errors::fluent::#slug);
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
                    #diag.emit();
                };

                implementation
            } else {
                span_err(
                    ast.span().unwrap(),
                    "`#[derive(LintDiagnostic)]` can only be used on structs",
                )
                .emit();

                DiagnosticDeriveError::ErrorHandled.to_compile_error()
            }
        };

        let diag = &builder.diag;
        structure.gen_impl(quote! {
            gen impl<'__a> rustc_errors::DecorateLint<'__a, ()> for @Self {
                fn decorate_lint(self, #diag: rustc_errors::LintDiagnosticBuilder<'__a, ()>) {
                    use rustc_errors::IntoDiagnosticArg;
                    #implementation
                }
            }
        })
    }
}
