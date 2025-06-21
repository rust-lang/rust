use std::io::{BufWriter, Write};

use anyhow::{Result, bail};
use camino::{Utf8Path, Utf8PathBuf};
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
    path: String,
    errors: Vec<Error>,
}

#[derive(Parser)]
struct Cli {
    /// The path to the json file to be linted
    json_path: String,

    postcard_path: String,

    /// Show verbose output
    #[arg(long)]
    verbose: bool,

    #[arg(long)]
    json_output: Option<String>,
}

fn main() -> Result<()> {
    let Cli { json_path, postcard_path, verbose, json_output } = Cli::parse();

    let json_path = normalize_path(&json_path);
    let postcard_path = normalize_path(&postcard_path);

    let contents = fs::read_to_string(&json_path)?;
    let krate: Crate = serde_json::from_str(&contents)?;
    assert_eq!(krate.format_version, FORMAT_VERSION);

    {
        let postcard_contents = fs::read(&postcard_path)?;

        use rustdoc_json_types::postcard::{MAGIC, Magic};

        assert_eq!(postcard_contents[..MAGIC.len()], MAGIC, "missing magic");

        let (magic, format_version, postcard_crate): (Magic, u32, Crate) =
            postcard::from_bytes(&postcard_contents)?;

        assert_eq!(magic, MAGIC);
        assert_eq!(format_version, FORMAT_VERSION);

        if postcard_crate != krate {
            bail!("{postcard_path} doesn't equal {json_path}");
        }
    }

    let krate_json: Value = serde_json::from_str(&contents)?;

    let mut validator = validator::Validator::new(&krate, krate_json);
    validator.check_crate();

    if let Some(json_output) = json_output {
        let output = JsonOutput { path: json_path.to_string(), errors: validator.errs.clone() };
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
        bail!("Errors validating json {json_path}");
    }

    Ok(())
}

fn normalize_path(path: &str) -> Utf8PathBuf {
    // We convert `-` into `_` for the file name to be sure the JSON path will always be correct.

    let path = Utf8Path::new(&path);
    let filename = path.file_name().unwrap().to_string().replace('-', "_");
    let parent = path.parent().unwrap();
    let path = parent.join(&filename);
    path
}
