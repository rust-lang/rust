//! This module generate AST datatype used by rust-analyzer.
//!
//! Specifically, it generates the `SyntaxKind` enum and a number of newtype
//! wrappers around `SyntaxNode` which implement `ra_syntax::AstNode`.

use std::{collections::BTreeMap, fs};

use proc_macro2::{Punct, Spacing};
use quote::{format_ident, quote};
use ron;
use serde::Deserialize;

use crate::{
    codegen::{self, update, Mode},
    project_root, Result,
};

pub fn generate_syntax(mode: Mode) -> Result<()> {
    let grammar = project_root().join(codegen::GRAMMAR);
    let grammar: Grammar = {
        let text = fs::read_to_string(grammar)?;
        ron::de::from_str(&text)?
    };

    let syntax_kinds_file = project_root().join(codegen::SYNTAX_KINDS);
    let syntax_kinds = generate_syntax_kinds(&grammar)?;
    update(syntax_kinds_file.as_path(), &syntax_kinds, mode)?;

    let ast_file = project_root().join(codegen::AST);
    let ast = generate_ast(&grammar)?;
    update(ast_file.as_path(), &ast, mode)?;

    Ok(())
}

fn generate_ast(grammar: &Grammar) -> Result<String> {
    let nodes = grammar.ast.iter().map(|(name, ast_node)| {
        let variants =
            ast_node.variants.iter().map(|var| format_ident!("{}", var)).collect::<Vec<_>>();
        let name = format_ident!("{}", name);

        let adt = if variants.is_empty() {
            let kind = format_ident!("{}", to_upper_snake_case(&name.to_string()));
            quote! {
                #[derive(Debug, Clone, PartialEq, Eq, Hash)]
                pub struct #name {
                    pub(crate) syntax: SyntaxNode,
                }

                impl AstNode for #name {
                    fn can_cast(kind: SyntaxKind) -> bool {
                        match kind {
                            #kind => true,
                            _ => false,
                        }
                    }
                    fn cast(syntax: SyntaxNode) -> Option<Self> {
                        if Self::can_cast(syntax.kind()) { Some(Self { syntax }) } else { None }
                    }
                    fn syntax(&self) -> &SyntaxNode { &self.syntax }
                }
            }
        } else {
            let kinds = variants
                .iter()
                .map(|name| format_ident!("{}", to_upper_snake_case(&name.to_string())))
                .collect::<Vec<_>>();

            quote! {
                #[derive(Debug, Clone, PartialEq, Eq, Hash)]
                pub enum #name {
                    #(#variants(#variants),)*
                }

                #(
                impl From<#variants> for #name {
                    fn from(node: #variants) -> #name {
                        #name::#variants(node)
                    }
                }
                )*

                impl AstNode for #name {
                    fn can_cast(kind: SyntaxKind) -> bool {
                        match kind {
                            #(#kinds)|* => true,
                            _ => false,
                        }
                    }
                    fn cast(syntax: SyntaxNode) -> Option<Self> {
                        let res = match syntax.kind() {
                            #(
                            #kinds => #name::#variants(#variants { syntax }),
                            )*
                            _ => return None,
                        };
                        Some(res)
                    }
                    fn syntax(&self) -> &SyntaxNode {
                        match self {
                            #(
                            #name::#variants(it) => &it.syntax,
                            )*
                        }
                    }
                }
            }
        };

        let traits = ast_node.traits.iter().map(|trait_name| {
            let trait_name = format_ident!("{}", trait_name);
            quote!(impl ast::#trait_name for #name {})
        });

        let collections = ast_node.collections.iter().map(|(name, kind)| {
            let method_name = format_ident!("{}", name);
            let kind = format_ident!("{}", kind);
            quote! {
                pub fn #method_name(&self) -> AstChildren<#kind> {
                    AstChildren::new(&self.syntax)
                }
            }
        });

        let options = ast_node.options.iter().map(|attr| {
            let method_name = match attr {
                Attr::Type(t) => format_ident!("{}", to_lower_snake_case(&t)),
                Attr::NameType(n, _) => format_ident!("{}", n),
            };
            let ty = match attr {
                Attr::Type(t) | Attr::NameType(_, t) => format_ident!("{}", t),
            };
            quote! {
                pub fn #method_name(&self) -> Option<#ty> {
                    AstChildren::new(&self.syntax).next()
                }
            }
        });

        quote! {
            #adt

            #(#traits)*

            impl #name {
                #(#collections)*
                #(#options)*
            }
        }
    });

    let ast = quote! {
        use crate::{
            SyntaxNode, SyntaxKind::{self, *},
            ast::{self, AstNode, AstChildren},
        };

        #(#nodes)*
    };

    let pretty = codegen::reformat(ast)?;
    Ok(pretty)
}

fn generate_syntax_kinds(grammar: &Grammar) -> Result<String> {
    let (single_byte_tokens_values, single_byte_tokens): (Vec<_>, Vec<_>) = grammar
        .punct
        .iter()
        .filter(|(token, _name)| token.len() == 1)
        .map(|(token, name)| (token.chars().next().unwrap(), format_ident!("{}", name)))
        .unzip();

    let punctuation_values = grammar.punct.iter().map(|(token, _name)| {
        if "{}[]()".contains(token) {
            let c = token.chars().next().unwrap();
            quote! { #c }
        } else {
            let cs = token.chars().map(|c| Punct::new(c, Spacing::Joint));
            quote! { #(#cs)* }
        }
    });
    let punctuation =
        grammar.punct.iter().map(|(_token, name)| format_ident!("{}", name)).collect::<Vec<_>>();

    let full_keywords_values = &grammar.keywords;
    let full_keywords =
        full_keywords_values.iter().map(|kw| format_ident!("{}_KW", to_upper_snake_case(&kw)));

    let all_keywords_values =
        grammar.keywords.iter().chain(grammar.contextual_keywords.iter()).collect::<Vec<_>>();
    let all_keywords_idents = all_keywords_values.iter().map(|kw| format_ident!("{}", kw));
    let all_keywords = all_keywords_values
        .iter()
        .map(|name| format_ident!("{}_KW", to_upper_snake_case(&name)))
        .collect::<Vec<_>>();

    let literals =
        grammar.literals.iter().map(|name| format_ident!("{}", name)).collect::<Vec<_>>();

    let tokens = grammar.tokens.iter().map(|name| format_ident!("{}", name)).collect::<Vec<_>>();

    let nodes = grammar.nodes.iter().map(|name| format_ident!("{}", name)).collect::<Vec<_>>();

    let ast = quote! {
        #![allow(bad_style, missing_docs, unreachable_pub)]
        /// The kind of syntax node, e.g. `IDENT`, `USE_KW`, or `STRUCT_DEF`.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
        #[repr(u16)]
        pub enum SyntaxKind {
            // Technical SyntaxKinds: they appear temporally during parsing,
            // but never end up in the final tree
            #[doc(hidden)]
            TOMBSTONE,
            #[doc(hidden)]
            EOF,
            #(#punctuation,)*
            #(#all_keywords,)*
            #(#literals,)*
            #(#tokens,)*
            #(#nodes,)*

            // Technical kind so that we can cast from u16 safely
            #[doc(hidden)]
            __LAST,
        }
        use self::SyntaxKind::*;

        impl SyntaxKind {
            pub fn is_keyword(self) -> bool {
                match self {
                    #(#all_keywords)|* => true,
                    _ => false,
                }
            }

            pub fn is_punct(self) -> bool {
                match self {
                    #(#punctuation)|* => true,
                    _ => false,
                }
            }

            pub fn is_literal(self) -> bool {
                match self {
                    #(#literals)|* => true,
                    _ => false,
                }
            }

            pub fn from_keyword(ident: &str) -> Option<SyntaxKind> {
                let kw = match ident {
                    #(#full_keywords_values => #full_keywords,)*
                    _ => return None,
                };
                Some(kw)
            }

            pub fn from_char(c: char) -> Option<SyntaxKind> {
                let tok = match c {
                    #(#single_byte_tokens_values => #single_byte_tokens,)*
                    _ => return None,
                };
                Some(tok)
            }
        }

        #[macro_export]
        macro_rules! T {
            #((#punctuation_values) => { $crate::SyntaxKind::#punctuation };)*
            #((#all_keywords_idents) => { $crate::SyntaxKind::#all_keywords };)*
        }
    };

    codegen::reformat(ast)
}

#[derive(Deserialize, Debug)]
struct Grammar {
    punct: Vec<(String, String)>,
    keywords: Vec<String>,
    contextual_keywords: Vec<String>,
    literals: Vec<String>,
    tokens: Vec<String>,
    nodes: Vec<String>,
    ast: BTreeMap<String, AstNode>,
}

#[derive(Deserialize, Debug)]
struct AstNode {
    #[serde(default)]
    #[serde(rename = "enum")]
    variants: Vec<String>,

    #[serde(default)]
    traits: Vec<String>,
    #[serde(default)]
    collections: Vec<(String, String)>,
    #[serde(default)]
    options: Vec<Attr>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum Attr {
    Type(String),
    NameType(String, String),
}

fn to_upper_snake_case(s: &str) -> String {
    let mut buf = String::with_capacity(s.len());
    let mut prev_is_upper = None;
    for c in s.chars() {
        if c.is_ascii_uppercase() && prev_is_upper == Some(false) {
            buf.push('_')
        }
        prev_is_upper = Some(c.is_ascii_uppercase());

        buf.push(c.to_ascii_uppercase());
    }
    buf
}

fn to_lower_snake_case(s: &str) -> String {
    let mut buf = String::with_capacity(s.len());
    let mut prev_is_upper = None;
    for c in s.chars() {
        if c.is_ascii_uppercase() && prev_is_upper == Some(false) {
            buf.push('_')
        }
        prev_is_upper = Some(c.is_ascii_uppercase());

        buf.push(c.to_ascii_lowercase());
    }
    buf
}
