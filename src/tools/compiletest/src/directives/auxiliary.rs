//! Code for dealing with test directives that request an "auxiliary" crate to
//! be built and made available to the test in some way.

use std::iter;

use camino::Utf8Path;

use super::directives::{AUX_BIN, AUX_BUILD, AUX_CODEGEN_BACKEND, AUX_CRATE, PROC_MACRO};
use crate::common::Config;

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
    pub(crate) crates: Vec<(String, String)>,
    /// Same as `builds`, but for proc-macros.
    pub(crate) proc_macros: Vec<String>,
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
            .chain(crates.iter().map(|(_, path)| path.as_str()))
            .chain(proc_macros.iter().map(String::as_str))
            .chain(codegen_backend.iter().map(String::as_str))
    }
}

/// If the given test directive line contains an `aux-*` directive, parse it
/// and update [`AuxProps`] accordingly.
pub(super) fn parse_and_update_aux(
    config: &Config,
    ln: &str,
    testfile: &Utf8Path,
    line_number: usize,
    aux: &mut AuxProps,
) {
    if !(ln.starts_with("aux-") || ln.starts_with("proc-macro")) {
        return;
    }

    config.push_name_value_directive(ln, AUX_BUILD, testfile, line_number, &mut aux.builds, |r| {
        r.trim().to_string()
    });
    config.push_name_value_directive(ln, AUX_BIN, testfile, line_number, &mut aux.bins, |r| {
        r.trim().to_string()
    });
    config.push_name_value_directive(
        ln,
        AUX_CRATE,
        testfile,
        line_number,
        &mut aux.crates,
        parse_aux_crate,
    );
    config.push_name_value_directive(
        ln,
        PROC_MACRO,
        testfile,
        line_number,
        &mut aux.proc_macros,
        |r| r.trim().to_string(),
    );
    if let Some(r) =
        config.parse_name_value_directive(ln, AUX_CODEGEN_BACKEND, testfile, line_number)
    {
        aux.codegen_backend = Some(r.trim().to_owned());
    }
}

fn parse_aux_crate(r: String) -> (String, String) {
    let mut parts = r.trim().splitn(2, '=');
    (
        parts.next().expect("missing aux-crate name (e.g. log=log.rs)").to_string(),
        parts.next().expect("missing aux-crate value (e.g. log=log.rs)").to_string(),
    )
}
