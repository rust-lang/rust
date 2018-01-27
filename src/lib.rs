extern crate unicode_xid;

mod text;
mod tree;
mod lexer;
mod parser;

#[cfg_attr(rustfmt, rustfmt_skip)]
pub mod syntax_kinds;
pub use text::{TextRange, TextUnit};
pub use tree::{File, FileBuilder, Node, Sink, SyntaxKind, Token};
pub use lexer::{next_token, tokenize};
pub use parser::parse;

pub mod utils {
    use std::fmt::Write;

    use {File, Node};

    pub fn dump_tree(file: &File) -> String {
        let mut result = String::new();
        go(file.root(), &mut result, 0);
        return result;

        fn go(node: Node, buff: &mut String, level: usize) {
            buff.push_str(&String::from("  ").repeat(level));
            write!(buff, "{:?}\n", node).unwrap();
            let my_errors = node.errors().filter(|e| e.after_child().is_none());
            let parent_errors = node.parent()
                .into_iter()
                .flat_map(|n| n.errors())
                .filter(|e| e.after_child() == Some(node));

            for err in my_errors {
                buff.push_str(&String::from("  ").repeat(level));
                write!(buff, "err: `{}`\n", err.message()).unwrap();
            }

            for child in node.children() {
                go(child, buff, level + 1)
            }

            for err in parent_errors {
                buff.push_str(&String::from("  ").repeat(level));
                write!(buff, "err: `{}`\n", err.message()).unwrap();
            }
        }
    }
}
