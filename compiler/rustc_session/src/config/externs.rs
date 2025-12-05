//! This module contains code to help parse and manipulate `--extern` arguments.

use std::path::PathBuf;

use rustc_errors::{Diag, FatalAbort};

use super::UnstableOptions;
use crate::EarlyDiagCtxt;

#[cfg(test)]
mod tests;

/// Represents the pieces of an `--extern` argument.
pub(crate) struct ExternOpt {
    pub(crate) crate_name: String,
    pub(crate) path: Option<PathBuf>,
    pub(crate) options: Option<String>,
}

/// Breaks out the major components of an `--extern` argument.
///
/// The options field will be a string containing comma-separated options that will need further
/// parsing and processing.
pub(crate) fn split_extern_opt<'a>(
    early_dcx: &'a EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    extern_opt: &str,
) -> Result<ExternOpt, Diag<'a, FatalAbort>> {
    let (name, path) = match extern_opt.split_once('=') {
        None => (extern_opt.to_string(), None),
        Some((name, path)) => (name.to_string(), Some(PathBuf::from(path))),
    };
    let (options, crate_name) = match name.split_once(':') {
        None => (None, name),
        Some((opts, crate_name)) => {
            if unstable_opts.namespaced_crates && crate_name.starts_with(':') {
                // If the name starts with `:`, we know this was actually something like `foo::bar` and
                // not a set of options. We can just use the original name as the crate name.
                (None, name)
            } else {
                (Some(opts.to_string()), crate_name.to_string())
            }
        }
    };

    if !valid_crate_name(&crate_name, unstable_opts) {
        let mut error = early_dcx.early_struct_fatal(format!(
            "crate name `{crate_name}` passed to `--extern` is not a valid ASCII identifier"
        ));
        let adjusted_name = crate_name.replace('-', "_");
        if is_ascii_ident(&adjusted_name) {
            #[allow(rustc::diagnostic_outside_of_impl)] // FIXME
            error
                .help(format!("consider replacing the dashes with underscores: `{adjusted_name}`"));
        }
        return Err(error);
    }

    Ok(ExternOpt { crate_name, path, options })
}

fn valid_crate_name(name: &str, unstable_opts: &UnstableOptions) -> bool {
    match name.split_once("::") {
        Some((a, b)) if unstable_opts.namespaced_crates => is_ascii_ident(a) && is_ascii_ident(b),
        Some(_) => false,
        None => is_ascii_ident(name),
    }
}

fn is_ascii_ident(string: &str) -> bool {
    let mut chars = string.chars();
    if let Some(start) = chars.next()
        && (start.is_ascii_alphabetic() || start == '_')
    {
        chars.all(|char| char.is_ascii_alphanumeric() || char == '_')
    } else {
        false
    }
}
