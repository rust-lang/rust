//! Code for dealing with test directives that request an "auxiliary" crate to
//! be built and made available to the test in some way.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::{fs, io, iter};

use super::{DirectiveLine, iter_header};
use crate::common::Config;
use crate::header::directives::{AUX_BIN, AUX_BUILD, AUX_CODEGEN_BACKEND, AUX_CRATE, PROC_MACRO};

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
pub(super) fn parse_and_update_aux(config: &Config, ln: &str, aux: &mut AuxProps) {
    if !(ln.starts_with("aux-") || ln.starts_with("proc-macro")) {
        return;
    }

    config.push_name_value_directive(ln, AUX_BUILD, &mut aux.builds, |r| r.trim().to_string());
    config.push_name_value_directive(ln, AUX_BIN, &mut aux.bins, |r| r.trim().to_string());
    config.push_name_value_directive(ln, AUX_CRATE, &mut aux.crates, parse_aux_crate);
    config
        .push_name_value_directive(ln, PROC_MACRO, &mut aux.proc_macros, |r| r.trim().to_string());
    if let Some(r) = config.parse_name_value_directive(ln, AUX_CODEGEN_BACKEND) {
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

/// Return an error if the given directory has cyclic aux.
pub(crate) fn check_cycles(config: &Config, dir: &Path) -> io::Result<()> {
    let mut filenames = vec![];
    let mut auxiliaries = HashMap::new();

    build_graph(config, dir, dir, &mut filenames, &mut auxiliaries)?;

    has_cycle(&filenames, &auxiliaries)
}

fn build_graph(
    config: &Config,
    dir: &Path,
    base_dir: &Path,
    filenames: &mut Vec<String>,
    auxiliaries: &mut HashMap<String, Vec<String>>,
) -> io::Result<()> {
    for file in fs::read_dir(dir)? {
        let file = file?;
        let file_path = file.path();

        if file_path.is_dir() {
            // explore in sub directory.
            build_graph(config, &file_path, base_dir, filenames, auxiliaries)?;
        } else {
            // We'd like to put a filename with relative path from the auxiliary directory (e.g., ["foo.rs", "foo/bar.rs"]).
            let relative_filename = file_path
                .strip_prefix(base_dir)
                .map_err(|e| io::Error::other(e))?
                .to_str()
                .unwrap();

            filenames.push(relative_filename.to_string());

            let mut aux_props = AuxProps::default();
            let mut poisoned = false;
            let f = File::open(&file_path).expect("open file to parse aux for cycle detection");
            iter_header(
                config.mode,
                &config.suite,
                &mut poisoned,
                &file_path,
                f,
                &mut |DirectiveLine { raw_directive: ln, .. }| {
                    parse_and_update_aux(config, ln, &mut aux_props);
                },
            );

            let mut auxs = vec![];
            for aux in aux_props.all_aux_path_strings() {
                auxs.push(aux.to_string());
            }

            if auxs.len() > 0 {
                auxiliaries.insert(relative_filename.to_string(), auxs);
            }
        }
    }

    Ok(())
}

/// has_cycle checks if the given graph has cycle.
/// It performs with a simple Depth-first search.
fn has_cycle(
    filenames: &Vec<String>,
    auxiliaries: &HashMap<String, Vec<String>>,
) -> io::Result<()> {
    // checked tracks nodes which the function already finished to search.
    let mut checked = HashSet::with_capacity(filenames.len());
    // During onde DFS exploration, on_search tracks visited nodes.
    // If the current node is already in on_search, that's a cycle.
    // The capacity `4` is added, because we can guess that an aux dependency is not so a long path.
    let mut on_search = HashSet::with_capacity(4);
    // path tracks visited nodes in on exploration.
    // This is used for generating an error message when a cycle is detected.
    let mut path = Vec::with_capacity(4);

    for vertex in filenames.iter() {
        if !checked.contains(vertex) {
            search(filenames, auxiliaries, &vertex, &mut checked, &mut on_search, &mut path)?;
        }
    }

    fn search(
        filenames: &Vec<String>,
        auxiliaries: &HashMap<String, Vec<String>>,
        vertex: &str,
        checked: &mut HashSet<String>,
        on_search: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> io::Result<()> {
        if !on_search.insert(vertex.to_string()) {
            let mut cyclic_path = vec![vertex];
            for v in path.iter().rev() {
                if v == vertex {
                    break;
                }
                cyclic_path.push(v);
            }

            return Err(io::Error::other(format!("detect cyclic auxiliary: {:?}", cyclic_path)));
        }

        if checked.insert(vertex.to_string()) {
            path.push(vertex.to_string());
            if let Some(auxs) = auxiliaries.get(&vertex.to_string()) {
                for aux in auxs.iter() {
                    search(filenames, auxiliaries, &aux, checked, on_search, path)?;
                }
            }
            path.pop().unwrap();
        }

        on_search.remove(&vertex.to_string());
        Ok(())
    }

    Ok(())
}
