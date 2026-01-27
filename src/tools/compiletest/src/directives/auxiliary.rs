//! Code for dealing with test directives that request an "auxiliary" crate to
//! be built and made available to the test in some way.

use std::iter;

use super::directives::{AUX_BIN, AUX_BUILD, AUX_CODEGEN_BACKEND, AUX_CRATE, PROC_MACRO};
use crate::common::Config;
use crate::directives::DirectiveLine;
use crate::util::static_regex;

#[cfg(test)]
mod tests;

/// The value of an `aux-crate` directive.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuxCrate {
    /// Contains `--extern` modifiers, if any. See the tracking issue for more
    /// info: <https://github.com/rust-lang/rust/issues/98405>
    /// With `aux-crate: noprelude:foo=bar.rs` this will be `noprelude`.
    pub extern_modifiers: Option<String>,
    /// With `aux-crate: foo=bar.rs` this will be `foo`.
    /// With `aux-crate: noprelude:foo=bar.rs` this will be `foo`.
    pub name: String,
    /// With `aux-crate: foo=bar.rs` this will be `bar.rs`.
    pub path: String,
}

/// The value of a `proc-macro` directive.
#[derive(Clone, Debug, Default)]
pub(crate) struct ProcMacro {
    /// With `proc-macro: bar.rs` this will be `bar.rs`.
    pub path: String,
}

/// Properties parsed from `aux-*` test directives.
#[derive(Clone, Debug, Default)]
pub(crate) struct AuxProps {
    /// Other crates that should be built and made available to this test.
    /// These are filenames relative to `./auxiliary/` in the test's directory.
    pub(crate) builds: Vec<String>,
    /// Auxiliary crates that should be compiled as `#![crate_type = "bin"]`.
    pub(crate) bins: Vec<String>,
    /// Similar to `builds`, but a list of NAME=somelib.rs of dependencies
    /// to build and pass with the `--extern` flag.
    pub(crate) crates: Vec<AuxCrate>,
    /// Same as `builds`, but for proc-macros.
    pub(crate) proc_macros: Vec<ProcMacro>,
    /// Similar to `builds`, but also uses the resulting dylib as a
    /// `-Zcodegen-backend` when compiling the test file.
    pub(crate) codegen_backend: Option<String>,
}

impl AuxProps {
    /// Yields all of the paths (relative to `./auxiliary/`) that have been
    /// specified in `aux-*` directives for this test.
    pub(crate) fn all_aux_path_strings(&self) -> impl Iterator<Item = &str> {
        let Self { builds, bins, crates, proc_macros, codegen_backend } = self;

        iter::empty()
            .chain(builds.iter().map(String::as_str))
            .chain(bins.iter().map(String::as_str))
            .chain(crates.iter().map(|c| c.path.as_str()))
            .chain(proc_macros.iter().map(|p| p.path.as_str()))
            .chain(codegen_backend.iter().map(String::as_str))
    }
}

/// If the given test directive line contains an `aux-*` directive, parse it
/// and update [`AuxProps`] accordingly.
pub(super) fn parse_and_update_aux(
    config: &Config,
    directive_line: &DirectiveLine<'_>,
    aux: &mut AuxProps,
) {
    if !(directive_line.name.starts_with("aux-") || directive_line.name == "proc-macro") {
        return;
    }

    let ln = directive_line;

    config.push_name_value_directive(ln, AUX_BUILD, &mut aux.builds, |r| r.trim().to_string());
    config.push_name_value_directive(ln, AUX_BIN, &mut aux.bins, |r| r.trim().to_string());
    config.push_name_value_directive(ln, AUX_CRATE, &mut aux.crates, parse_aux_crate);
    config.push_name_value_directive(ln, PROC_MACRO, &mut aux.proc_macros, parse_proc_macro);

    if let Some(r) = config.parse_name_value_directive(ln, AUX_CODEGEN_BACKEND) {
        aux.codegen_backend = Some(r.trim().to_owned());
    }
}

fn parse_aux_crate(r: String) -> AuxCrate {
    let r = r.trim();

    // Matches:
    //   name=path
    //   modifiers:name=path
    let caps = static_regex!(r"^(?:(?<modifiers>[^=]*?):)?(?<name>[^=]*)=(?<path>.*)$")
        .captures(r)
        .unwrap_or_else(|| {
            panic!("couldn't parse aux-crate value `{r}` (should be e.g. `log=log.rs`)")
        });

    let modifiers = caps.name("modifiers").map(|m| m.as_str().to_string());
    let name = caps["name"].to_string();
    let path = caps["path"].to_string();

    AuxCrate { extern_modifiers: modifiers, name, path }
}

fn parse_proc_macro(r: String) -> ProcMacro {
    ProcMacro { path: r.trim().to_string() }
}
