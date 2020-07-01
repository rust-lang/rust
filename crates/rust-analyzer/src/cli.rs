//! Various batch processing tasks, intended primarily for debugging.

mod load_cargo;
mod analysis_stats;
mod analysis_bench;
mod diagnostics;
mod progress_report;
mod ssr;

use std::io::Read;

use anyhow::Result;
use ra_ide::{file_structure, Analysis};
use ra_prof::profile;
use ra_syntax::{AstNode, SourceFile};

pub use analysis_bench::{analysis_bench, BenchWhat, Position};
pub use analysis_stats::analysis_stats;
pub use diagnostics::diagnostics;
pub use load_cargo::load_cargo;
pub use ssr::{apply_ssr_rules, search_for_patterns};

#[derive(Clone, Copy)]
pub enum Verbosity {
    Spammy,
    Verbose,
    Normal,
    Quiet,
}

impl Verbosity {
    pub fn is_verbose(self) -> bool {
        matches!(self, Verbosity::Verbose | Verbosity::Spammy)
    }
    pub fn is_spammy(self) -> bool {
        matches!(self, Verbosity::Spammy)
    }
}

pub fn parse(no_dump: bool) -> Result<()> {
    let _p = profile("parsing");
    let file = file()?;
    if !no_dump {
        println!("{:#?}", file.syntax());
    }
    std::mem::forget(file);
    Ok(())
}

pub fn symbols() -> Result<()> {
    let file = file()?;
    for s in file_structure(&file) {
        println!("{:?}", s);
    }
    Ok(())
}

pub fn highlight(rainbow: bool) -> Result<()> {
    let (analysis, file_id) = Analysis::from_single_file(read_stdin()?);
    let html = analysis.highlight_as_html(file_id, rainbow).unwrap();
    println!("{}", html);
    Ok(())
}

fn file() -> Result<SourceFile> {
    let text = read_stdin()?;
    Ok(SourceFile::parse(&text).tree())
}

fn read_stdin() -> Result<String> {
    let mut buff = String::new();
    std::io::stdin().read_to_string(&mut buff)?;
    Ok(buff)
}
