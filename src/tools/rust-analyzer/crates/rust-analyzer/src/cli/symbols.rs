//! Read Rust code on stdin, print syntax tree on stdout.
use ide::Analysis;

use crate::cli::{flags, read_stdin};

impl flags::Symbols {
    pub fn run(self) -> anyhow::Result<()> {
        let text = read_stdin()?;
        let (analysis, file_id) = Analysis::from_single_file(text);
        let structure = analysis.file_structure(file_id).unwrap();
        for s in structure {
            println!("{s:?}");
        }
        Ok(())
    }
}
