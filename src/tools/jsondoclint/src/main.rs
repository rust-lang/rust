use std::env;

use anyhow::{anyhow, bail, Result};
use fs_err as fs;
use rustdoc_json_types::{Crate, Id, FORMAT_VERSION};
use serde_json::Value;

pub(crate) mod item_kind;
mod json_find;
mod validator;

#[derive(Debug)]
struct Error {
    kind: ErrorKind,
    id: Id,
}

#[derive(Debug)]
enum ErrorKind {
    NotFound,
    Custom(String),
}

fn main() -> Result<()> {
    let path = env::args().nth(1).ok_or_else(|| anyhow!("no path given"))?;
    let contents = fs::read_to_string(&path)?;
    let krate: Crate = serde_json::from_str(&contents)?;
    assert_eq!(krate.format_version, FORMAT_VERSION);

    let mut validator = validator::Validator::new(&krate);
    validator.check_crate();

    if !validator.errs.is_empty() {
        for err in validator.errs {
            match err.kind {
                ErrorKind::NotFound => {
                    let krate_json: Value = serde_json::from_str(&contents)?;

                    let sels =
                        json_find::find_selector(&krate_json, &Value::String(err.id.0.clone()));
                    match &sels[..] {
                        [] => unreachable!(
                            "id must be in crate, or it wouldn't be reported as not found"
                        ),
                        [sel] => eprintln!(
                            "{} not in index or paths, but refered to at '{}'",
                            err.id.0,
                            json_find::to_jsonpath(&sel)
                        ),
                        [sel, ..] => eprintln!(
                            "{} not in index or paths, but refered to at '{}' and more",
                            err.id.0,
                            json_find::to_jsonpath(&sel)
                        ),
                    }
                }
                ErrorKind::Custom(msg) => eprintln!("{}: {}", err.id.0, msg),
            }
        }
        bail!("Errors validating json {path}");
    }

    Ok(())
}
