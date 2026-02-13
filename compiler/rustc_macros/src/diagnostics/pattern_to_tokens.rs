use fluent_syntax::ast::{
    Expression, InlineExpression, NamedArgument, Pattern, PatternElement, Variant, VariantKey,
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::LitStr;

pub(crate) fn pattern_to_tokens(pattern: &Pattern<&str>, span: Span) -> TokenStream {
    let elements = pattern.elements.iter().map(|elem| generate_pattern_elem(elem, span));
    quote! {
        rustc_errors::Pattern {
            elements: vec![
                #(#elements),*
            ]
        }
    }
}

fn generate_pattern_elem(elem: &PatternElement<&str>, span: Span) -> TokenStream {
    match elem {
        PatternElement::TextElement { value } => {
            let lit = generate_name(value, span);
            quote! { rustc_errors::PatternElement::TextElement { value: #lit } }
        }
        PatternElement::Placeable { expression } => {
            let expr = generate_expr(expression, span);
            quote! { rustc_errors::PatternElement::Placeable { expression: #expr } }
        }
    }
}

fn generate_expr(expr: &Expression<&str>, span: Span) -> TokenStream {
    match expr {
        Expression::Select { selector, variants } => {
            let selector = generate_inline_expr(selector, span);
            let variants = variants.iter().map(|v| generate_variant(v, span));
            quote! { rustc_errors::Expression::Select {
                selector: #selector,
                variants: vec![#(#variants),*]
            } }
        }
        Expression::Inline(expr) => {
            let expr = generate_inline_expr(expr, span);
            quote! { rustc_errors::Expression::Inline(#expr) }
        }
    }
}

fn generate_variant(variant: &Variant<&str>, span: Span) -> TokenStream {
    let key = generate_variant_key(&variant.key, span);
    let value = pattern_to_tokens(&variant.value, span);
    let default = variant.default;
    quote! { rustc_errors::Variant {
        key: #key,
        value: #value,
        default: #default,
    } }
}

fn generate_variant_key(key: &VariantKey<&str>, span: Span) -> TokenStream {
    match key {
        VariantKey::Identifier { name } => {
            let name = generate_name(name, span);
            quote!(rustc_errors::VariantKey::Identifier{ name: #name })
        }
        VariantKey::NumberLiteral { value } => {
            let value = generate_name(value, span);
            quote!(rustc_errors::VariantKey::NumberLiteral{ value: #value })
        }
    }
}

fn generate_inline_expr(expr: &InlineExpression<&str>, span: Span) -> TokenStream {
    match expr {
        InlineExpression::VariableReference { id } => {
            let lit = generate_name(id.name, span);
            quote! { rustc_errors::InlineExpression::VariableReference { id: rustc_errors::Identifier { name: #lit } } }
        }
        InlineExpression::StringLiteral { value } => {
            let lit = generate_name(value, span);
            quote! { rustc_errors::InlineExpression::StringLiteral { value: #lit } }
        }
        InlineExpression::FunctionReference { id, arguments } => {
            let name = generate_name(id.name, span);
            let positional = arguments.positional.iter().map(|arg| generate_inline_expr(arg, span));
            let named = arguments.named.iter().map(|arg| generate_named_argument(arg, span));

            quote! { rustc_errors::InlineExpression::FunctionReference {
                id: rustc_errors::Identifier { name: #name },
                arguments: rustc_errors::CallArguments {
                    positional: vec![#(#positional),*],
                    named: vec![#(#named),*],
                }
            } }
        }
        _ => panic!("Found unrenderable fluent expr: {expr:?}"),
    }
}

fn generate_named_argument(arg: &NamedArgument<&str>, span: Span) -> TokenStream {
    let name = generate_name(arg.name.name, span);
    let value = generate_inline_expr(&arg.value, span);
    quote! { rustc_errors::NamedArgument {
        name: rustc_errors::Identifier { name: #name }, value: #value
    }  }
}

fn generate_name(name: &str, span: Span) -> TokenStream {
    let name = LitStr::new(name, span);
    quote! { #name }
}
