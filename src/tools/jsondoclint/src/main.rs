use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use clap::Parser;
use fs_err as fs;
use rustdoc_json_types::{Crate, Id, FORMAT_VERSION};
use serde::Serialize;
use serde_json::Value;

pub(crate) mod item_kind;
mod json_find;
mod validator;

#[derive(Debug, PartialEq, Eq, Serialize, Clone)]
struct Error {
    kind: ErrorKind,
    id: Id,
}

#[derive(Debug, PartialEq, Eq, Serialize, Clone)]
enum ErrorKind {
    NotFound(Vec<json_find::Selector>),
    Custom(String),
}

#[derive(Debug, Serialize)]
struct JsonOutput {
    path: PathBuf,
    errors: Vec<Error>,
}

#[derive(Parser)]
struct Cli {
    /// The path to the json file to be linted
    path: String,

    /// Show verbose output
    #[arg(long)]
    verbose: bool,

    #[arg(long)]
    json_output: Option<String>,
}

fn main() -> Result<()> {
    let Cli { path, verbose, json_output } = Cli::parse();

    // We convert `-` into `_` for the file name to be sure the JSON path will always be correct.
    let path = Path::new(&path);
    let filename = path.file_name().unwrap().to_str().unwrap().replace('-', "_");
    let parent = path.parent().unwrap();
    let path = parent.join(&filename);

    let contents = fs::read_to_string(&path)?;
    let krate: Crate = serde_json::from_str(&contents)?;
    assert_eq!(krate.format_version, FORMAT_VERSION);

    let krate_json: Value = serde_json::from_str(&contents)?;

    let mut validator = validator::Validator::new(&krate, krate_json);
    validator.check_crate();

    if let Some(json_output) = json_output {
        let output = JsonOutput { path: path.clone(), errors: validator.errs.clone() };
        let mut f = BufWriter::new(fs::File::create(json_output)?);
        serde_json::to_writer(&mut f, &output)?;
        f.flush()?;
    }

    if !validator.errs.is_empty() {
        for err in validator.errs {
            match err.kind {
                ErrorKind::NotFound(sels) => match &sels[..] {
                    [] => {
                        unreachable!(
                            "id {:?} must be in crate, or it wouldn't be reported as not found",
                            err.id
                        )
                    }
                    [sel] => eprintln!(
                        "{} not in index or paths, but referred to at '{}'",
                        err.id.0,
                        json_find::to_jsonpath(&sel)
                    ),
                    [sel, ..] => {
                        if verbose {
                            let sels = sels
                                .iter()
                                .map(json_find::to_jsonpath)
                                .map(|i| format!("'{i}'"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            eprintln!(
                                "{} not in index or paths, but referred to at {sels}",
                                err.id.0
                            );
                        } else {
                            eprintln!(
                                "{} not in index or paths, but referred to at '{}' and {} more",
                                err.id.0,
                                json_find::to_jsonpath(&sel),
                                sels.len() - 1,
                            )
                        }
                    }
                },
                ErrorKind::Custom(msg) => eprintln!("{}: {}", err.id.0, msg),
            }
        }
        bail!("Errors validating json {}", path.display());
    }

    Ok(())
}
