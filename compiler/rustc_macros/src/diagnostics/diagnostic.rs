#![deny(unused_must_use)]

use std::cell::RefCell;

use crate::diagnostics::diagnostic_builder::{DiagnosticDeriveBuilder, DiagnosticDeriveKind};
use crate::diagnostics::error::{span_err, DiagnosticDeriveError};
use crate::diagnostics::utils::SetOnce;
use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use synstructure::Structure;

/// The central struct for constructing the `into_diagnostic` method from an annotated struct.
pub(crate) struct DiagnosticDerive<'a> {
    structure: Structure<'a>,
    builder: DiagnosticDeriveBuilder,
}

impl<'a> DiagnosticDerive<'a> {
    pub(crate) fn new(diag: syn::Ident, handler: syn::Ident, structure: Structure<'a>) -> Self {
        Self {
            builder: DiagnosticDeriveBuilder {
                diag,
                kind: DiagnosticDeriveKind::Diagnostic { handler },
            },
            structure,
        }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let DiagnosticDerive { mut structure, mut builder } = self;

        let slugs = RefCell::new(Vec::new());
        let implementation = builder.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble(variant);
            let body = builder.body(variant);

            let diag = &builder.parent.diag;
            let DiagnosticDeriveKind::Diagnostic { handler } = &builder.parent.kind else {
                unreachable!()
            };
            let init = match builder.slug.value_ref() {
                None => {
                    span_err(builder.span, "diagnostic slug not specified")
                        .help("specify the slug as the first argument to the `#[diag(...)]` \
                            attribute, such as `#[diag(hir_analysis_example_error)]`")
                        .emit();
                    return DiagnosticDeriveError::ErrorHandled.to_compile_error();
                }
                Some(slug) if let Some( Mismatch { slug_name, crate_name, slug_prefix }) = Mismatch::check(slug) => {
                    span_err(slug.span().unwrap(), "diagnostic slug and crate name do not match")
                        .note(format!(
                            "slug is `{slug_name}` but the crate name is `{crate_name}`"
                        ))
                        .help(format!(
                            "expected a slug starting with `{slug_prefix}_...`"
                        ))
                        .emit();
                    return DiagnosticDeriveError::ErrorHandled.to_compile_error();
                }
                Some(slug) => {
                    slugs.borrow_mut().push(slug.clone());
                    quote! {
                        let mut #diag = #handler.struct_diagnostic(crate::fluent_generated::#slug);
                    }
                }
            };

            let formatting_init = &builder.formatting_init;
            quote! {
                #init
                #formatting_init
                #preamble
                #body
                #diag
            }
        });

        let DiagnosticDeriveKind::Diagnostic { handler } = &builder.kind else { unreachable!() };

        let mut imp = structure.gen_impl(quote! {
            gen impl<'__diagnostic_handler_sess, G>
                    rustc_errors::IntoDiagnostic<'__diagnostic_handler_sess, G>
                    for @Self
                where G: rustc_errors::EmissionGuarantee
            {

                #[track_caller]
                fn into_diagnostic(
                    self,
                    #handler: &'__diagnostic_handler_sess rustc_errors::Handler
                ) -> rustc_errors::DiagnosticBuilder<'__diagnostic_handler_sess, G> {
                    use rustc_errors::IntoDiagnosticArg;
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
    builder: DiagnosticDeriveBuilder,
}

impl<'a> LintDiagnosticDerive<'a> {
    pub(crate) fn new(diag: syn::Ident, structure: Structure<'a>) -> Self {
        Self {
            builder: DiagnosticDeriveBuilder { diag, kind: DiagnosticDeriveKind::LintDiagnostic },
            structure,
        }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let LintDiagnosticDerive { mut structure, mut builder } = self;

        let implementation = builder.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble(variant);
            let body = builder.body(variant);

            let diag = &builder.parent.diag;
            let formatting_init = &builder.formatting_init;
            quote! {
                #preamble
                #formatting_init
                #body
                #diag
            }
        });

        let slugs = RefCell::new(Vec::new());
        let msg = builder.each_variant(&mut structure, |mut builder, variant| {
            // Collect the slug by generating the preamble.
            let _ = builder.preamble(variant);

            match builder.slug.value_ref() {
                None => {
                    span_err(builder.span, "diagnostic slug not specified")
                        .help("specify the slug as the first argument to the attribute, such as \
                            `#[diag(compiletest_example)]`")
                        .emit();
                    DiagnosticDeriveError::ErrorHandled.to_compile_error()
                }
                Some(slug) if let Some( Mismatch { slug_name, crate_name, slug_prefix }) = Mismatch::check(slug) => {
                    span_err(slug.span().unwrap(), "diagnostic slug and crate name do not match")
                        .note(format!(
                            "slug is `{slug_name}` but the crate name is `{crate_name}`"
                        ))
                        .help(format!(
                            "expected a slug starting with `{slug_prefix}_...`"
                        ))
                        .emit();
                    DiagnosticDeriveError::ErrorHandled.to_compile_error()
                }
                Some(slug) => {
                    slugs.borrow_mut().push(slug.clone());
                    quote! {
                        crate::fluent_generated::#slug.into()
                    }
                }
            }
        });

        let diag = &builder.diag;
        let mut imp = structure.gen_impl(quote! {
            gen impl<'__a> rustc_errors::DecorateLint<'__a, ()> for @Self {
                #[track_caller]
                fn decorate_lint<'__b>(
                    self,
                    #diag: &'__b mut rustc_errors::DiagnosticBuilder<'__a, ()>
                ) -> &'__b mut rustc_errors::DiagnosticBuilder<'__a, ()> {
                    use rustc_errors::IntoDiagnosticArg;
                    #implementation
                }

                fn msg(&self) -> rustc_errors::DiagnosticMessage {
                    #msg
                }
            }
        });
        for test in slugs.borrow().iter().map(|s| generate_test(s, &structure)) {
            imp.extend(test);
        }

        imp
    }
}

struct Mismatch {
    slug_name: String,
    crate_name: String,
    slug_prefix: String,
}

impl Mismatch {
    /// Checks whether the slug starts with the crate name it's in.
    fn check(slug: &syn::Path) -> Option<Mismatch> {
        // If this is missing we're probably in a test, so bail.
        let crate_name = std::env::var("CARGO_CRATE_NAME").ok()?;

        // If we're not in a "rustc_" crate, bail.
        let Some(("rustc", slug_prefix)) = crate_name.split_once('_') else { return None };

        let slug_name = slug.segments.first()?.ident.to_string();
        if !slug_name.starts_with(slug_prefix) {
            Some(Mismatch { slug_name, slug_prefix: slug_prefix.to_string(), crate_name })
        } else {
            None
        }
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
    // We need to make sure that the same diagnostic slug can be used multiple times without causing an
    // error, so just have a global counter here.
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
