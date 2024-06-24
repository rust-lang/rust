//! Read Rust code on stdin, print syntax tree on stdout.
use ide::Edition;
use syntax::{AstNode, SourceFile};

use crate::cli::{flags, read_stdin};

impl flags::Parse {
    pub fn run(self) -> anyhow::Result<()> {
        let _p = tracing::info_span!("flags::Parse::run").entered();
        let text = read_stdin()?;
        let file = SourceFile::parse(&text, Edition::CURRENT).tree();
        if !self.no_dump {
            println!("{:#?}", file.syntax());
        }
        std::mem::forget(file);
        Ok(())
    }
}
