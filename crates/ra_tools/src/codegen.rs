use std::{collections::BTreeMap, fs, path::Path};

use quote::quote;
use ron;
use serde::Deserialize;

use crate::{project_root, Mode, Result, AST, GRAMMAR};

pub fn generate(mode: Mode) -> Result<()> {
    let grammar = project_root().join(GRAMMAR);
    // let syntax_kinds = project_root().join(SYNTAX_KINDS);
    let ast = project_root().join(AST);
    generate_ast(&grammar, &ast, mode)
}

fn generate_ast(grammar_src: &Path, dst: &Path, mode: Mode) -> Result<()> {
    let src: Grammar = {
        let text = fs::read_to_string(grammar_src)?;
        ron::de::from_str(&text)?
    };
    eprintln!("{:#?}", src);
    Ok(())
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
    traits: Vec<String>,
    #[serde(default)]
    collections: Vec<Attr>,
    #[serde(default)]
    options: Vec<Attr>,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum Attr {
    Type(String),
    NameType(String, String),
}
