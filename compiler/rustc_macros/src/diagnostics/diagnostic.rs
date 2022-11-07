#![deny(unused_must_use)]

use crate::diagnostics::diagnostic_builder::{DiagnosticDeriveBuilder, DiagnosticDeriveKind};
use crate::diagnostics::error::{span_err, DiagnosticDeriveError};
use crate::diagnostics::utils::SetOnce;
use proc_macro2::TokenStream;
use quote::quote;
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

        let implementation = builder.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble(&variant);
            let body = builder.body(&variant);

            let diag = &builder.parent.diag;
            let DiagnosticDeriveKind::Diagnostic { handler } = &builder.parent.kind else {
                unreachable!()
            };
            let init = match builder.slug.value_ref() {
                None => {
                    span_err(builder.span, "diagnostic slug not specified")
                        .help(&format!(
                            "specify the slug as the first argument to the `#[diag(...)]` \
                            attribute, such as `#[diag(hir_analysis_example_error)]`",
                        ))
                        .emit();
                    return DiagnosticDeriveError::ErrorHandled.to_compile_error();
                }
                Some(slug) => {
                    let check = make_check(slug);
                    quote! {
                        #check
                        let mut #diag = #handler.struct_diagnostic(rustc_errors::fluent::#slug);
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
        structure.gen_impl(quote! {
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
            builder: DiagnosticDeriveBuilder { diag, kind: DiagnosticDeriveKind::LintDiagnostic },
            structure,
        }
    }

    pub(crate) fn into_tokens(self) -> TokenStream {
        let LintDiagnosticDerive { mut structure, mut builder } = self;

        let implementation = builder.each_variant(&mut structure, |mut builder, variant| {
            let preamble = builder.preamble(&variant);
            let body = builder.body(&variant);

            let diag = &builder.parent.diag;
            let formatting_init = &builder.formatting_init;
            quote! {
                #preamble
                #formatting_init
                #body
                #diag
            }
        });

        let msg = builder.each_variant(&mut structure, |mut builder, variant| {
            // Collect the slug by generating the preamble.
            let _ = builder.preamble(&variant);

            match builder.slug.value_ref() {
                None => {
                    span_err(builder.span, "diagnostic slug not specified")
                        .help(&format!(
                            "specify the slug as the first argument to the attribute, such as \
                            `#[diag(compiletest_example)]`",
                        ))
                        .emit();
                    return DiagnosticDeriveError::ErrorHandled.to_compile_error();
                }
                Some(slug) => {
                    let check = make_check(slug);

                    quote! {
                        #check
                        rustc_errors::fluent::#slug.into()
                    }
                }
            }
        });

        let diag = &builder.diag;
        structure.gen_impl(quote! {
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
        })
    }
}

/// Checks whether the slug starts with the crate name it's in.
fn make_check(slug: &syn::Path) -> TokenStream {
    quote! {
        const _: () = {
            let krate = env!("CARGO_MANIFEST_DIR").as_bytes();

            let mut start = 0;
            while !(krate[start] == b'r'
                && krate[start + 1] == b'u'
                && krate[start + 2] == b's'
                && krate[start + 3] == b't'
                && krate[start + 4] == b'c'
                && krate[start + 5] == b'_')
            {
                if krate.len() == start + 5 {
                    panic!(concat!("crate does not contain \"rustc_\": ", env!("CARGO_MANIFEST_DIR")));
                }
                start += 1;
            }
            start += 6;

            let slug = stringify!(#slug).as_bytes();

            let mut pos = 0;
            loop {
                let b = slug[pos];
                if krate.len() == start + pos {
                    if b != b'_' {
                        panic!(concat!("slug \"", stringify!(#slug), "\" does not match the crate (", env!("CARGO_MANIFEST_DIR") ,") it is in"));
                    }
                    break
                }
                let a = krate[start+pos];

                if a != b {
                    panic!(concat!("slug \"", stringify!(#slug), "\" does not match the crate (", env!("CARGO_MANIFEST_DIR") ,") it is in"));
                }
                pos += 1;
            }
        };
    }
}
