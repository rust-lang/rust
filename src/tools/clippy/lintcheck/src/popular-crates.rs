#![deny(clippy::pedantic)]

use clap::Parser;
use crates_io_api::{CratesQueryBuilder, Sort, SyncClient};
use indicatif::ProgressBar;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
struct Opts {
    /// Output TOML file name
    output: PathBuf,
    /// Number of crate names to download
    #[clap(short, long, default_value_t = 100)]
    number: usize,
    /// Do not output progress
    #[clap(short, long)]
    quiet: bool,
}

fn main() -> anyhow::Result<()> {
    let opts = Opts::parse();
    let mut output = BufWriter::new(File::create(opts.output)?);
    output.write_all(b"[crates]\n")?;
    let client = SyncClient::new(
        "clippy/lintcheck (github.com/rust-lang/rust-clippy/)",
        Duration::from_secs(1),
    )?;
    let mut seen_crates = HashSet::new();
    let pb = if opts.quiet {
        None
    } else {
        Some(ProgressBar::new(opts.number as u64))
    };
    let mut query = CratesQueryBuilder::new()
        .sort(Sort::RecentDownloads)
        .page_size(100)
        .build();
    while seen_crates.len() < opts.number {
        let retrieved = client.crates(query.clone())?.crates;
        if retrieved.is_empty() {
            eprintln!("No more than {} crates available from API", seen_crates.len());
            break;
        }
        for c in retrieved {
            if seen_crates.insert(c.name.clone()) {
                output.write_all(
                    format!(
                        "{} = {{ name = '{}', versions = ['{}'] }}\n",
                        c.name, c.name, c.max_version
                    )
                    .as_bytes(),
                )?;
                if let Some(pb) = &pb {
                    pb.inc(1);
                }
            }
        }
        query.set_page(query.page() + 1);
    }
    Ok(())
}
