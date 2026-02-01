use fluent_bundle::FluentResource;
use fluent_syntax::ast::{Expression, InlineExpression, Pattern, PatternElement};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Path;
use synstructure::{Structure, VariantInfo};

use crate::diagnostics::error::span_err;

#[derive(Clone)]
pub(crate) enum Message {
    Slug(Path),
    Inline(Span, String),
}

impl Message {
    pub(crate) fn diag_message(&self, variant: &VariantInfo<'_>) -> TokenStream {
        match self {
            Message::Slug(slug) => {
                quote! { crate::fluent_generated::#slug }
            }
            Message::Inline(message_span, message) => {
                verify_fluent_message(*message_span, &message, variant);
                quote! { rustc_errors::DiagMessage::Inline(std::borrow::Cow::Borrowed(#message)) }
            }
        }
    }

    /// Generates a `#[test]` that verifies that all referenced variables
    /// exist on this structure.
    pub(crate) fn generate_test(&self, structure: &Structure<'_>) -> TokenStream {
        match self {
            Message::Slug(slug) => {
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
                let ident = quote::format_ident!(
                    "verify_{slug}_{}",
                    COUNTER.fetch_add(1, Ordering::Relaxed)
                );
                let ref_slug = quote::format_ident!("{slug}_refs");
                let struct_name = &structure.ast().ident;
                let variables: Vec<_> = structure
                    .variants()
                    .iter()
                    .flat_map(|v| {
                        v.ast()
                            .fields
                            .iter()
                            .filter_map(|f| f.ident.as_ref().map(|i| i.to_string()))
                    })
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
            Message::Inline(..) => {
                // We don't generate a test for inline diagnostics, we can verify these at compile-time!
                // This verification is done in the `diag_message` function above
                quote! {}
            }
        }
    }
}

fn verify_fluent_message(msg_span: Span, message: &str, variant: &VariantInfo<'_>) {
    // Parse the fluent message
    const GENERATED_MSG_ID: &str = "generated_msg";
    let resource = FluentResource::try_new(format!("{GENERATED_MSG_ID} = {message}\n")).unwrap();
    assert_eq!(resource.entries().count(), 1);
    let Some(fluent_syntax::ast::Entry::Message(message)) = resource.get_entry(0) else {
        panic!("Did not parse into a message")
    };

    // Check if all variables are used
    let fields: Vec<String> = variant
        .bindings()
        .iter()
        .flat_map(|b| b.ast().ident.as_ref())
        .map(|id| id.to_string())
        .collect();
    for variable in variable_references(&message) {
        if !fields.iter().any(|f| f == variable) {
            span_err(msg_span.unwrap(), format!("Variable `{variable}` not found in diagnostic "))
                .help(format!("Available fields: {:?}", fields.join(", ")))
                .emit();
        }
        // assert!(, );
    }
}

fn variable_references<'a>(msg: &fluent_syntax::ast::Message<&'a str>) -> Vec<&'a str> {
    let mut refs = vec![];
    if let Some(Pattern { elements }) = &msg.value {
        for elt in elements {
            if let PatternElement::Placeable {
                expression: Expression::Inline(InlineExpression::VariableReference { id }),
            } = elt
            {
                refs.push(id.name);
            }
        }
    }
    for attr in &msg.attributes {
        for elt in &attr.value.elements {
            if let PatternElement::Placeable {
                expression: Expression::Inline(InlineExpression::VariableReference { id }),
            } = elt
            {
                refs.push(id.name);
            }
        }
    }
    refs
}
