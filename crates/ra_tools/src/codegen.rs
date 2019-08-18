use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

use heck::{ShoutySnakeCase, SnakeCase};
use quote::{format_ident, quote};
use ron;
use serde::Deserialize;

use crate::{project_root, Mode, Result, AST, GRAMMAR, SYNTAX_KINDS};

pub fn generate(mode: Mode) -> Result<()> {
    let grammar = project_root().join(GRAMMAR);
    let grammar: Grammar = {
        let text = fs::read_to_string(grammar)?;
        ron::de::from_str(&text)?
    };

    let _syntax_kinds = project_root().join(SYNTAX_KINDS);
    let _ast = project_root().join(AST);

    let ast = generate_ast(&grammar)?;
    println!("{}", ast);
    Ok(())
}

fn generate_ast(grammar: &Grammar) -> Result<String> {
    let nodes = grammar.ast.iter().map(|(name, ast_node)| {
        let variants =
            ast_node.variants.iter().map(|var| format_ident!("{}", var)).collect::<Vec<_>>();
        let name = format_ident!("{}", name);

        let kinds = if variants.is_empty() { vec![name.clone()] } else { variants.clone() }
            .into_iter()
            .map(|name| format_ident!("{}", name.to_string().to_shouty_snake_case()))
            .collect::<Vec<_>>();

        let variants = if variants.is_empty() {
            None
        } else {
            let kind_enum = format_ident!("{}Kind", name);
            Some(quote!(
                pub enum #kind_enum {
                    #(#variants(#variants),)*
                }

                #(
                impl From<#variants> for #name {
                    fn from(node: #variants) -> #name {
                        #name { syntax: node.syntax }
                    }
                }
                )*

                impl #name {
                    pub fn kind(&self) -> #kind_enum {
                        let syntax = self.syntax.clone();
                        match syntax.kind() {
                            #(
                            #kinds =>
                                #kind_enum::#variants(#variants { syntax }),
                            )*
                            _ => unreachable!(),
                        }
                    }
                }
            ))
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
                Attr::Type(t) => format_ident!("{}", t.to_snake_case()),
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
            #[derive(Debug, Clone, PartialEq, Eq, Hash)]
            pub struct #name {
                pub(crate) syntax: SyntaxNode,
            }

            impl AstNode for #name {
                fn can_cast(kind: SyntaxKind) -> bool {
                    match kind {
                        #(#kinds)|* => true,
                        _ => false,
                    }
                }
                fn cast(syntax: SyntaxNode) -> Option<Self> {
                    if Self::can_cast(syntax.kind()) { Some(Self { syntax }) } else { None }
                }
                fn syntax(&self) -> &SyntaxNode { &self.syntax }
            }

            #variants

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

    let pretty = reformat(ast)?;
    Ok(pretty)
}

fn reformat(text: impl std::fmt::Display) -> Result<String> {
    let mut rustfmt = Command::new("rustfmt")
        .arg("--config-path")
        .arg(project_root().join("rustfmt.toml"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    write!(rustfmt.stdin.take().unwrap(), "{}", text)?;
    let output = rustfmt.wait_with_output()?;
    let stdout = String::from_utf8(output.stdout)?;
    let preamble = "Generated file, do not edit by hand, see `crate/ra_tools/src/codegen`";
    Ok(format!("// {}\n\n{}", preamble, stdout))
}

#[derive(Deserialize, Debug)]
struct Grammar {
    single_byte_tokens: Vec<(String, String)>,
    multi_byte_tokens: Vec<(String, String)>,
    keywords: Vec<String>,
    contextual_keywords: Vec<String>,
    literals: Vec<String>,
    tokens: Vec<String>,
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
