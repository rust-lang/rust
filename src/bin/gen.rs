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
    syntax_kinds: Vec<String>,
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
        for (idx, kind) in self.syntax_kinds.iter().enumerate() {
            let sname = scream(kind);
            write!(
                acc,
                "pub const {}: SyntaxKind = SyntaxKind({});\n",
                sname, idx
            ).unwrap();
        }
        acc.push_str("\n");
        for kind in self.syntax_kinds.iter() {
            let sname = scream(kind);
            write!(
                acc,
                "static {sname}_INFO: SyntaxInfo = SyntaxInfo {{\n   name: \"{sname}\",\n}};\n",
                sname = sname
            ).unwrap();
        }
        acc.push_str("\n");

        acc.push_str("pub(crate) fn syntax_info(kind: SyntaxKind) -> &'static SyntaxInfo {\n");
        acc.push_str("    match kind {\n");
        for kind in self.syntax_kinds.iter() {
            let sname = scream(kind);
            write!(
                acc,
                "        {sname} => &{sname}_INFO,\n",
                sname = sname
            ).unwrap();
        }
        acc.push_str("        _ => unreachable!()\n");
        acc.push_str("    }\n");
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