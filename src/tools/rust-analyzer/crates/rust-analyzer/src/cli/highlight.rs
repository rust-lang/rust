//! Read Rust code on stdin, print HTML highlighted version to stdout.

use ide::Analysis;

use crate::cli::{flags, read_stdin};

impl flags::Highlight {
    pub fn run(self) -> anyhow::Result<()> {
        let (analysis, file_id) = Analysis::from_single_file(read_stdin()?);
        let html = analysis.highlight_as_html(file_id, self.rainbow).unwrap();
        println!("{html}");
        Ok(())
    }
}
