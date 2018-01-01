extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate ron;
extern crate file;

use std::path::PathBuf;
use std::ascii::AsciiExt;
use std::fmt::Write;

fn main() {
    let grammar = Grammar::read();
    let text = grammar.to_syntax_kinds();
    file::put_text(&generated_file(), &text).unwrap();
}

#[derive(Deserialize)]
struct Grammar {
    keywords: Vec<String>,
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
        acc.push_str("// Generated from grammar.ron\n");
        acc.push_str("use tree::{SyntaxKind, SyntaxInfo};\n");
        acc.push_str("\n");

        let syntax_kinds: Vec<String> =
            self.keywords.iter().map(|kw| kw_token(kw))
                .chain(self.tokens.iter().cloned())
                .chain(self.nodes.iter().cloned())
                .collect();

        for (idx, kind) in syntax_kinds.iter().enumerate() {
            let sname = scream(kind);
            write!(
                acc,
                "pub const {}: SyntaxKind = SyntaxKind({});\n",
                sname, idx
            ).unwrap();
        }
        acc.push_str("\n");
        write!(acc, "static INFOS: [SyntaxInfo; {}] = [\n", syntax_kinds.len()).unwrap();
        for kind in syntax_kinds.iter() {
            let sname = scream(kind);
            write!(
                acc,
                "    SyntaxInfo {{ name: \"{sname}\" }},\n",
                sname = sname
            ).unwrap();
        }
        acc.push_str("];\n");
        acc.push_str("\n");

        acc.push_str("pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {\n");
        acc.push_str("    &INFOS[kind.0 as usize]\n");
        acc.push_str("}\n\n");
        acc.push_str("pub(crate) fn ident_to_keyword(ident: &str) -> Option<SyntaxKind> {\n");
        acc.push_str("   match ident {\n");
        for kw in self.keywords.iter() {
            write!(acc, "       {:?} => Some({}),\n", kw, kw_token(kw)).unwrap();
        }
        acc.push_str("       _ => None,\n");
        acc.push_str("   }\n");
        acc.push_str("}\n");
        acc
    }
}

fn grammar_file() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("grammar.ron")
}

fn generated_file() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("src/syntax_kinds.rs")
}

fn scream(word: &str) -> String {
    word.chars().map(|c| c.to_ascii_uppercase()).collect()
}

fn kw_token(keyword: &str) -> String {
    format!("{}_KW", scream(keyword))
}