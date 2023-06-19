//! Various batch processing tasks, intended primarily for debugging.

pub mod flags;
pub mod load_cargo;
mod parse;
mod symbols;
mod highlight;
mod analysis_stats;
mod diagnostics;
mod ssr;
mod lsif;
mod scip;

mod progress_report;

use std::io::Read;

use anyhow::Result;
use ide::AnalysisHost;
use vfs::Vfs;

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

fn read_stdin() -> Result<String> {
    let mut buff = String::new();
    std::io::stdin().read_to_string(&mut buff)?;
    Ok(buff)
}

fn report_metric(metric: &str, value: u64, unit: &str) {
    if std::env::var("RA_METRICS").is_err() {
        return;
    }
    println!("METRIC:{metric}:{value}:{unit}")
}

fn print_memory_usage(mut host: AnalysisHost, vfs: Vfs) {
    let mem = host.per_query_memory_usage();

    let before = profile::memory_usage();
    drop(vfs);
    let vfs = before.allocated - profile::memory_usage().allocated;

    let before = profile::memory_usage();
    drop(host);
    let unaccounted = before.allocated - profile::memory_usage().allocated;
    let remaining = profile::memory_usage().allocated;

    for (name, bytes, entries) in mem {
        // NOTE: Not a debug print, so avoid going through the `eprintln` defined above.
        eprintln!("{bytes:>8} {entries:>6} {name}");
    }
    eprintln!("{vfs:>8}        VFS");

    eprintln!("{unaccounted:>8}        Unaccounted");

    eprintln!("{remaining:>8}        Remaining");
}
