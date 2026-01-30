#![deny(unused_must_use)]

use std::cell::RefCell;

use proc_macro2::TokenStream;
use quote::quote;
use synstructure::Structure;

use crate::diagnostics::diagnostic_builder::DiagnosticDeriveKind;
use crate::diagnostics::error::DiagnosticDeriveError;

/// The central struct for constructing the `into_diag` method from an annotated struct.
pub(crate) struct DiagnosticDerive<'a> {
    structure: Structure<'a>,
}

impl<'a> DiagnosticDerive<'a> {
    pub(crate) fn new(structure: Structure<'a>) -> Self {
        Self { structure }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let DiagnosticDerive { mut structure } = self;
        let kind = DiagnosticDeriveKind::Diagnostic;
        let messages = RefCell::new(Vec::new());
        let implementation = kind.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble(variant);
            let body = builder.body(variant);

            let Some(message) = builder.primary_message() else {
                return DiagnosticDeriveError::ErrorHandled.to_compile_error();
            };
            messages.borrow_mut().push(message.clone());
            let message = message.diag_message(variant);

            let init = quote! {
                let mut diag = rustc_errors::Diag::new(
                    dcx,
                    level,
                    #message
                );
            };

            let formatting_init = &builder.formatting_init;
            quote! {
                #init
                #formatting_init
                #preamble
                #body
                diag
            }
        });

        // A lifetime of `'a` causes conflicts, but `_sess` is fine.
        // FIXME(edition_2024): Fix the `keyword_idents_2024` lint to not trigger here?
        #[allow(keyword_idents_2024)]
        let mut imp = structure.gen_impl(quote! {
            gen impl<'_sess, G> rustc_errors::Diagnostic<'_sess, G> for @Self
                where G: rustc_errors::EmissionGuarantee
            {
                #[track_caller]
                fn into_diag(
                    self,
                    dcx: rustc_errors::DiagCtxtHandle<'_sess>,
                    level: rustc_errors::Level
                ) -> rustc_errors::Diag<'_sess, G> {
                    #implementation
                }
            }
        });
        for test in messages.borrow().iter().map(|s| s.generate_test(&structure)) {
            imp.extend(test);
        }
        imp
    }
}

/// The central struct for constructing the `decorate_lint` method from an annotated struct.
pub(crate) struct LintDiagnosticDerive<'a> {
    structure: Structure<'a>,
}

impl<'a> LintDiagnosticDerive<'a> {
    pub(crate) fn new(structure: Structure<'a>) -> Self {
        Self { structure }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let LintDiagnosticDerive { mut structure } = self;
        let kind = DiagnosticDeriveKind::LintDiagnostic;
        let messages = RefCell::new(Vec::new());
        let implementation = kind.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble(variant);
            let body = builder.body(variant);

            let Some(message) = builder.primary_message() else {
                return DiagnosticDeriveError::ErrorHandled.to_compile_error();
            };
            messages.borrow_mut().push(message.clone());
            let message = message.diag_message(variant);
            let primary_message = quote! {
                diag.primary_message(#message);
            };

            let formatting_init = &builder.formatting_init;
            quote! {
                #primary_message
                #preamble
                #formatting_init
                #body
                diag
            }
        });

        // FIXME(edition_2024): Fix the `keyword_idents_2024` lint to not trigger here?
        #[allow(keyword_idents_2024)]
        let mut imp = structure.gen_impl(quote! {
            gen impl<'__a> rustc_errors::LintDiagnostic<'__a, ()> for @Self {
                #[track_caller]
                fn decorate_lint<'__b>(
                    self,
                    diag: &'__b mut rustc_errors::Diag<'__a, ()>
                ) {
                    #implementation;
                }
            }
        });
        for test in messages.borrow().iter().map(|s| s.generate_test(&structure)) {
            imp.extend(test);
        }

        imp
    }
}
