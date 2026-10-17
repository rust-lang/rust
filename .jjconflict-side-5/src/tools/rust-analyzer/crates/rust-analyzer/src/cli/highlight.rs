//! Read Rust code on stdin, print HTML highlighted version to stdout.

use ide::Analysis;
use ide_db::base_db::AbsPathBuf;
use triomphe::Arc;

use crate::cli::{flags, read_stdin};

impl flags::Highlight {
    pub fn run(self) -> anyhow::Result<()> {
        let cwd = AbsPathBuf::assert_utf8(std::env::current_dir()?);
        let (analysis, file_id) = Analysis::from_single_file(read_stdin()?, Arc::new(cwd));
        let html = analysis.highlight_as_html(file_id, self.rainbow).unwrap();
        println!("{html}");
        Ok(())
    }
}
