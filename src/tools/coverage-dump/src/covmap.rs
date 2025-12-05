use std::collections::HashMap;
use std::sync::LazyLock;

use anyhow::{Context, ensure};
use regex::Regex;

use crate::llvm_utils::{truncated_md5, unescape_llvm_string_contents};
use crate::parser::Parser;

#[derive(Debug, Default)]
pub(crate) struct FilenameTables {
    map: HashMap<u64, Vec<String>>,
}

impl FilenameTables {
    pub(crate) fn lookup(&self, filenames_hash: u64, global_file_id: usize) -> Option<&str> {
        let table = self.map.get(&filenames_hash)?;
        let filename = table.get(global_file_id)?;
        Some(filename)
    }
}

struct CovmapLineData {
    payload: Vec<u8>,
}

pub(crate) fn make_filename_tables(llvm_ir: &str) -> anyhow::Result<FilenameTables> {
    let mut map = HashMap::default();

    for line in llvm_ir.lines().filter(|line| is_covmap_line(line)) {
        let CovmapLineData { payload } = parse_covmap_line(line)?;

        let mut parser = Parser::new(&payload);
        let n_filenames = parser.read_uleb128_usize()?;
        let uncompressed_bytes = parser.read_chunk_to_uncompressed_bytes()?;
        parser.ensure_empty()?;

        let mut filenames_table = vec![];

        let mut parser = Parser::new(&uncompressed_bytes);
        for _ in 0..n_filenames {
            let len = parser.read_uleb128_usize()?;
            let bytes = parser.read_n_bytes(len)?;
            let filename = str::from_utf8(bytes)?;
            filenames_table.push(filename.to_owned());
        }

        let filenames_hash = truncated_md5(&payload);
        map.insert(filenames_hash, filenames_table);
    }

    Ok(FilenameTables { map })
}

fn is_covmap_line(line: &str) -> bool {
    line.starts_with("@__llvm_coverage_mapping ")
}

fn parse_covmap_line(line: &str) -> anyhow::Result<CovmapLineData> {
    ensure!(is_covmap_line(line));

    const RE_STRING: &str = r#"(?x)^
        @__llvm_coverage_mapping \ =
        .*
        \[ [0-9]+ \ x \ i8 \] \ c"(?<payload>[^"]*)"
        .*$
    "#;
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(RE_STRING).unwrap());

    let captures =
        RE.captures(line).with_context(|| format!("couldn't parse covmap line: {line:?}"))?;
    let payload = unescape_llvm_string_contents(&captures["payload"]);

    Ok(CovmapLineData { payload })
}
