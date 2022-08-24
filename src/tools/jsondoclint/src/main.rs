use std::env;

use anyhow::{anyhow, bail, Result};
use fs_err as fs;
use rustdoc_json_types::{Crate, Id, FORMAT_VERSION};

pub(crate) mod item_kind;
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
    let contents = fs::read_to_string(path)?;
    let krate: Crate = serde_json::from_str(&contents)?;
    assert_eq!(krate.format_version, FORMAT_VERSION);

    let mut validator = validator::Validator::new(&krate);
    validator.check_crate();

    if !validator.errs.is_empty() {
        for err in validator.errs {
            match err.kind {
                ErrorKind::NotFound => eprintln!("{}: Not Found", err.id.0),
                ErrorKind::Custom(msg) => eprintln!("{}: {}", err.id.0, msg),
            }
        }
        bail!("Errors validating json");
    }

    Ok(())
}
