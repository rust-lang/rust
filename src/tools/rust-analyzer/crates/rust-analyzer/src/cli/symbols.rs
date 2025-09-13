//! Read Rust code on stdin, print syntax tree on stdout.
use ide::{Analysis, FileStructureConfig};

use crate::cli::{flags, read_stdin};

impl flags::Symbols {
    pub fn run(self) -> anyhow::Result<()> {
        let text = read_stdin()?;
        let (analysis, file_id) = Analysis::from_single_file(text);
        let structure = analysis
            // The default setting in config.rs (document_symbol_search_excludeLocals) is to exclude
            // locals because it is unlikely that users want document search to return the names of
            // local variables, but here we include them deliberately.
            .file_structure(&FileStructureConfig { exclude_locals: false }, file_id)
            .unwrap();
        for s in structure {
            println!("{s:?}");
        }
        Ok(())
    }
}
