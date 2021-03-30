//! Various batch processing tasks, intended primarily for debugging.

pub(crate) mod load_cargo;
mod analysis_stats;
mod diagnostics;
mod progress_report;
mod ssr;

use std::io::Read;

use anyhow::Result;
use ide::{Analysis, AnalysisHost};
use syntax::{AstNode, SourceFile};
use vfs::Vfs;

pub use self::{
    analysis_stats::AnalysisStatsCmd,
    diagnostics::diagnostics,
    load_cargo::{load_workspace, load_workspace_at, LoadCargoConfig},
    ssr::{apply_ssr_rules, search_for_patterns},
};

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
    let _p = profile::span("parsing");
    let file = file()?;
    if !no_dump {
        println!("{:#?}", file.syntax());
    }
    std::mem::forget(file);
    Ok(())
}

pub fn symbols() -> Result<()> {
    let text = read_stdin()?;
    let (analysis, file_id) = Analysis::from_single_file(text);
    let structure = analysis.file_structure(file_id).unwrap();
    for s in structure {
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

fn report_metric(metric: &str, value: u64, unit: &str) {
    if std::env::var("RA_METRICS").is_err() {
        return;
    }
    println!("METRIC:{}:{}:{}", metric, value, unit)
}

fn print_memory_usage(mut host: AnalysisHost, vfs: Vfs) {
    let mut mem = host.per_query_memory_usage();

    let before = profile::memory_usage();
    drop(vfs);
    let vfs = before.allocated - profile::memory_usage().allocated;
    mem.push(("VFS".into(), vfs));

    let before = profile::memory_usage();
    drop(host);
    mem.push(("Unaccounted".into(), before.allocated - profile::memory_usage().allocated));

    mem.push(("Remaining".into(), profile::memory_usage().allocated));

    for (name, bytes) in mem {
        // NOTE: Not a debug print, so avoid going through the `eprintln` defined above.
        eprintln!("{:>8} {}", bytes, name);
    }
}
