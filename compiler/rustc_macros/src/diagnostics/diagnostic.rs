#![deny(unused_must_use)]

use proc_macro2::TokenStream;
use quote::quote;
use synstructure::Structure;

use crate::diagnostics::diagnostic_builder::each_variant;
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
        let implementation = each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble(variant);
            let body = builder.body(variant);

            let Some(message) = builder.primary_message() else {
                return DiagnosticDeriveError::ErrorHandled.to_compile_error();
            };
            let message = message.diag_message(Some(variant));

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
        structure.gen_impl(quote! {
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
        })
    }
}
