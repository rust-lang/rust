use fluent_bundle::FluentResource;
use fluent_syntax::ast::{Expression, InlineExpression, Pattern, PatternElement};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::ext::IdentExt;
use synstructure::VariantInfo;

use crate::diagnostics::error::span_err;

#[derive(Clone)]
pub(crate) struct Message {
    pub message_span: Span,
    pub value: String,
}

impl Message {
    /// Get the diagnostic message for this diagnostic
    /// The passed `variant` is used to check whether all variables in the message are used.
    /// For subdiagnostics, we cannot check this.
    pub(crate) fn diag_message(&self, variant: Option<&VariantInfo<'_>>) -> TokenStream {
        let message = &self.value;
        verify_fluent_message(self.message_span, &message, variant);
        quote! { rustc_errors::DiagMessage::Inline(std::borrow::Cow::Borrowed(#message)) }
    }
}

fn verify_fluent_message(msg_span: Span, message_str: &str, variant: Option<&VariantInfo<'_>>) {
    // Parse the fluent message
    const GENERATED_MSG_ID: &str = "generated_msg";
    let resource =
        FluentResource::try_new(format!("{GENERATED_MSG_ID} = {message_str}\n")).unwrap();
    assert_eq!(resource.entries().count(), 1);
    let Some(fluent_syntax::ast::Entry::Message(message)) = resource.get_entry(0) else {
        panic!("Did not parse into a message")
    };

    // Check if all variables are used
    if let Some(variant) = variant {
        let fields: Vec<String> = variant
            .bindings()
            .iter()
            .flat_map(|b| b.ast().ident.as_ref())
            .map(|id| id.unraw().to_string())
            .collect();
        for variable in variable_references(&message) {
            if !fields.iter().any(|f| f == variable) {
                span_err(
                    msg_span.unwrap(),
                    format!("Variable `{variable}` not found in diagnostic "),
                )
                .help(format!("Available fields: {:?}", fields.join(", ")))
                .emit();
            }
        }
    }

    verify_message_style(msg_span, message_str);
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

const ALLOWED_CAPITALIZED_WORDS: &[&str] = &[
    // tidy-alphabetical-start
    "ABI",
    "ABIs",
    "ADT",
    "C-variadic",
    "CGU-reuse",
    "Cargo",
    "Ferris",
    "GCC",
    "MIR",
    "NaNs",
    "OK",
    "Rust",
    "Unicode",
    "VS",
    // tidy-alphabetical-end
];

/// See: https://rustc-dev-guide.rust-lang.org/diagnostics.html#diagnostic-output-style-guide
fn verify_message_style(msg_span: Span, message: &str) {
    // Verify that message starts with lowercase char
    let Some(first_word) = message.split_whitespace().next() else {
        span_err(msg_span.unwrap(), "message must not be empty").emit();
        return;
    };
    let first_char = first_word.chars().next().expect("Word is not empty");
    if first_char.is_uppercase() && !ALLOWED_CAPITALIZED_WORDS.contains(&first_word) {
        span_err(msg_span.unwrap(), "message `{value}` starts with an uppercase letter. Fix it or add it to `ALLOWED_CAPITALIZED_WORDS`").emit();
        return;
    }

    // Verify that message does not end in `.`
    if message.ends_with(".") && !message.ends_with("...") {
        span_err(msg_span.unwrap(), "message `{value}` ends with a period").emit();
        return;
    }
}
