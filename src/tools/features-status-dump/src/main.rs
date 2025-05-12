use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use tidy::features::{Feature, collect_lang_features, collect_lib_features};

#[derive(Debug, Parser)]
struct Cli {
    /// Path to `library/` directory.
    #[arg(long)]
    library_path: PathBuf,
    /// Path to `compiler/` directory.
    #[arg(long)]
    compiler_path: PathBuf,
    /// Path to `output/` directory.
    #[arg(long)]
    output_path: PathBuf,
}

#[derive(Debug, serde::Serialize)]
struct FeaturesStatus {
    lang_features_status: HashMap<String, Feature>,
    lib_features_status: HashMap<String, Feature>,
}

fn main() -> Result<()> {
    let Cli { compiler_path, library_path, output_path } = Cli::parse();

    let lang_features_status = collect_lang_features(&compiler_path, &mut false);
    let lib_features_status = collect_lib_features(&library_path)
        .into_iter()
        .filter(|&(ref name, _)| !lang_features_status.contains_key(name))
        .collect();
    let features_status = FeaturesStatus { lang_features_status, lib_features_status };

    let output_dir = output_path.parent().with_context(|| {
        format!("failed to get parent dir of output path `{}`", output_path.display())
    })?;
    std::fs::create_dir_all(output_dir).with_context(|| {
        format!("failed to create output directory at `{}`", output_dir.display())
    })?;

    let output_file = File::create(&output_path).with_context(|| {
        format!("failed to create file at given output path `{}`", output_path.display())
    })?;
    let writer = BufWriter::new(output_file);
    serde_json::to_writer_pretty(writer, &features_status)
        .context("failed to write json output")?;
    Ok(())
}
