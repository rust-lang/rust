#![deny(unused_must_use)]

use std::cell::RefCell;

use proc_macro2::TokenStream;
use quote::quote;
use synstructure::Structure;

use crate::diagnostics::diagnostic_builder::DiagnosticDeriveKind;
use crate::diagnostics::parse_diag::PrimaryMessage;

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
        let slugs = RefCell::new(Vec::new());
        let implementation = kind.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble();
            let body = builder.body(variant);

            let PrimaryMessage::Slug(slug) = builder.diag_attr.primary_message;
            slugs.borrow_mut().push(slug.clone());
            let init = quote! {
                let mut diag = rustc_errors::Diag::new(
                    dcx,
                    level,
                    crate::fluent_generated::#slug
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
        for test in slugs.borrow().iter().map(|s| generate_test(s, &structure)) {
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
        let slugs = RefCell::new(Vec::new());
        let implementation = kind.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble();
            let body = builder.body(variant);

            let PrimaryMessage::Slug(slug) = builder.diag_attr.primary_message;
            slugs.borrow_mut().push(slug.clone());

            let primary_message = quote! {
                diag.primary_message(crate::fluent_generated::#slug);
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
        for test in slugs.borrow().iter().map(|s| generate_test(s, &structure)) {
            imp.extend(test);
        }

        imp
    }
}

/// Generates a `#[test]` that verifies that all referenced variables
/// exist on this structure.
fn generate_test(slug: &syn::Path, structure: &Structure<'_>) -> TokenStream {
    // FIXME: We can't identify variables in a subdiagnostic
    for field in structure.variants().iter().flat_map(|v| v.ast().fields.iter()) {
        for attr_name in field.attrs.iter().filter_map(|at| at.path().get_ident()) {
            if attr_name == "subdiagnostic" {
                return quote!();
            }
        }
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    // We need to make sure that the same diagnostic slug can be used multiple times without
    // causing an error, so just have a global counter here.
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let slug = slug.get_ident().unwrap();
    let ident = quote::format_ident!("verify_{slug}_{}", COUNTER.fetch_add(1, Ordering::Relaxed));
    let ref_slug = quote::format_ident!("{slug}_refs");
    let struct_name = &structure.ast().ident;
    let variables: Vec<_> = structure
        .variants()
        .iter()
        .flat_map(|v| v.ast().fields.iter().filter_map(|f| f.ident.as_ref().map(|i| i.to_string())))
        .collect();
    // tidy errors on `#[test]` outside of test files, so we use `#[test ]` to work around this
    quote! {
        #[cfg(test)]
        #[test ]
        fn #ident() {
            let variables = [#(#variables),*];
            for vref in crate::fluent_generated::#ref_slug {
                assert!(variables.contains(vref), "{}: variable `{vref}` not found ({})", stringify!(#struct_name), stringify!(#slug));
            }
        }
    }
}
