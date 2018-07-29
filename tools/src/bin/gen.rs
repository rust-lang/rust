extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate file;
extern crate ron;

use std::path::PathBuf;
use std::fmt::Write;

fn main() {
    let grammar = Grammar::read();
    let text = grammar.to_syntax_kinds();
    let target = generated_file();
    if text != file::get_text(&target).unwrap_or_default() {
        file::put_text(&target, &text).unwrap();
    }
}

#[derive(Deserialize)]
struct Grammar {
    keywords: Vec<String>,
    contextual_keywords: Vec<String>,
    tokens: Vec<String>,
    nodes: Vec<String>,
}

impl Grammar {
    fn read() -> Grammar {
        let text = file::get_text(&grammar_file()).unwrap();
        ron::de::from_str(&text).unwrap()
    }

    fn to_syntax_kinds(&self) -> String {
        let mut acc = String::new();
        acc.push_str("#![allow(bad_style, missing_docs, unreachable_pub)]\n");
        acc.push_str("#![cfg_attr(rustfmt, rustfmt_skip)]\n");
        acc.push_str("//! Generated from grammar.ron\n");
        acc.push_str("use super::SyntaxInfo;\n");
        acc.push_str("\n");

        let syntax_kinds: Vec<String> = self.tokens
            .iter()
            .cloned()
            .chain(self.keywords.iter().map(|kw| kw_token(kw)))
            .chain(self.contextual_keywords.iter().map(|kw| kw_token(kw)))
            .chain(self.nodes.iter().cloned())
            .collect();

        // enum SyntaxKind
        acc.push_str("/// The kind of syntax node, e.g. `IDENT`, `USE_KW`, or `STRUCT_DEF`.\n");
        acc.push_str("#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]\n");
        acc.push_str("pub enum SyntaxKind {\n");
        for kind in syntax_kinds.iter() {
            write!(acc, "    {},\n", scream(kind)).unwrap();
        }
        acc.push_str("\n");
        acc.push_str("    // Technical SyntaxKinds: they appear temporally during parsing,\n");
        acc.push_str("    // but never end up in the final tree\n");
        acc.push_str("    #[doc(hidden)]\n");
        acc.push_str("    TOMBSTONE,\n");
        acc.push_str("    #[doc(hidden)]\n");
        acc.push_str("    EOF,\n");
        acc.push_str("}\n");
        acc.push_str("pub(crate) use self::SyntaxKind::*;\n");
        acc.push_str("\n");

        // fn info
        acc.push_str("impl SyntaxKind {\n");
        acc.push_str("    pub(crate) fn info(self) -> &'static SyntaxInfo {\n");
        acc.push_str("        match self {\n");
        for kind in syntax_kinds.iter() {
            let sname = scream(kind);
            write!(
                acc,
                "            {sname} => &SyntaxInfo {{ name: \"{sname}\" }},\n",
                sname = sname
            ).unwrap();
        }
        acc.push_str("\n");
        acc.push_str("            TOMBSTONE => &SyntaxInfo { name: \"TOMBSTONE\" },\n");
        acc.push_str("            EOF => &SyntaxInfo { name: \"EOF\" },\n");
        acc.push_str("        }\n");
        acc.push_str("    }\n");

        // fn from_keyword
        acc.push_str("    pub(crate) fn from_keyword(ident: &str) -> Option<SyntaxKind> {\n");
        acc.push_str("        match ident {\n");
        // NB: no contextual_keywords here!
        for kw in self.keywords.iter() {
            write!(acc, "            {:?} => Some({}),\n", kw, kw_token(kw)).unwrap();
        }
        acc.push_str("            _ => None,\n");
        acc.push_str("        }\n");
        acc.push_str("    }\n");
        acc.push_str("}\n");
        acc.push_str("\n");
        acc
    }
}

fn grammar_file() -> PathBuf {
    base_dir().join("grammar.ron")
}

fn generated_file() -> PathBuf {
    base_dir().join("src/syntax_kinds/generated.rs")
}

fn scream(word: &str) -> String {
    word.chars().map(|c| c.to_ascii_uppercase()).collect()
}

fn kw_token(keyword: &str) -> String {
    format!("{}_KW", scream(keyword))
}

fn base_dir() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).parent().unwrap().to_owned()
}
