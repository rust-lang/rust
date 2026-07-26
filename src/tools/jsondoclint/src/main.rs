use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::Parser;
use fs_err as fs;
use rustdoc_json_types::{Crate, FORMAT_VERSION, Id};
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
    json_path: String,

    /// The path of the postcard file to be linted
    postcard_path: String,

    /// Show verbose output
    #[arg(long)]
    verbose: bool,

    #[arg(long)]
    json_output: Option<String>,
}

fn input_path(path: &str) -> PathBuf {
    // We convert `-` into `_` for the file name to be sure the JSON path will always be correct.
    let path = Path::new(&path);
    let filename = path.file_name().unwrap().to_str().unwrap().replace('-', "_");
    let parent = path.parent().unwrap();
    let path = parent.join(&filename);
    path
}

fn main() -> Result<()> {
    let Cli { json_path, postcard_path, verbose, json_output } = Cli::parse();
    let json_path = input_path(&json_path);
    let postcard_path = input_path(&postcard_path);

    let json_contents = fs::read_to_string(&json_path)?;

    let krate: Crate = serde_json::from_str(&json_contents)?;
    assert_eq!(krate.format_version, FORMAT_VERSION);

    check_postcard(&postcard_path, &krate)?;

    let krate_json: Value = serde_json::from_str(&json_contents)?;

    let mut validator = validator::Validator::new(&krate, krate_json);
    validator.check_crate();

    if let Some(json_output) = json_output {
        let output = JsonOutput { path: json_path.clone(), errors: validator.errs.clone() };
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
                        json_find::to_jsonpath(sel)
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
                                json_find::to_jsonpath(sel),
                                sels.len() - 1,
                            )
                        }
                    }
                },
                ErrorKind::Custom(msg) => eprintln!("{}: {}", err.id.0, msg),
            }
        }
        bail!("Errors validating json {}", json_path.display());
    }

    Ok(())
}

fn check_postcard(path: &Path, expected_krate: &Crate) -> Result<()> {
    let postcard_bytes = fs::read(path)?;

    let (file, rest) =
        postcard::take_from_bytes::<rustdoc_json_types::postcard::File>(&postcard_bytes)?;

    if !rest.is_empty() {
        bail!("Postcard file has {} leftover bytes", rest.len());
    }

    let (magic, format_version, krate) = file;

    let expected_magic = rustdoc_json_types::postcard::MAGIC;
    if magic != expected_magic {
        bail!("Postcard file has bad magic value, got {magic:?} but expected {expected_magic:?}");
    }

    let expected_format_version = rustdoc_json_types::FORMAT_VERSION;
    if format_version != expected_format_version {
        bail!(
            "Postcard file has bad format version, got {format_version} but expected {expected_format_version}"
        );
    }
    if &krate != expected_krate {
        bail!("Postcard file didn't contain same crate information as json file");
    }

    Ok(())
}
