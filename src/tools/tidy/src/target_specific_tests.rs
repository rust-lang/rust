//! Tidy check to ensure that all target specific tests (those that require a `--target` flag)
//! also require the pre-requisite LLVM components to run.

use std::collections::BTreeMap;
use std::path::Path;

use crate::iter_header::{HeaderLine, iter_header};
use crate::walk::filter_not_rust;

const LLVM_COMPONENTS_HEADER: &str = "needs-llvm-components:";
const COMPILE_FLAGS_HEADER: &str = "compile-flags:";

#[derive(Default, Debug)]
struct RevisionInfo<'a> {
    target_arch: Option<&'a str>,
    llvm_components: Option<Vec<&'a str>>,
}

pub fn check(tests_path: &Path, bad: &mut bool) {
    crate::walk::walk(tests_path, |path, _is_dir| filter_not_rust(path), &mut |entry, content| {
        let file = entry.path().display();
        let mut header_map = BTreeMap::new();
        iter_header(content, &mut |HeaderLine { revision, directive, .. }| {
            if let Some(value) = directive.strip_prefix(LLVM_COMPONENTS_HEADER) {
                let info = header_map.entry(revision).or_insert(RevisionInfo::default());
                let comp_vec = info.llvm_components.get_or_insert(Vec::new());
                for component in value.split(' ') {
                    let component = component.trim();
                    if !component.is_empty() {
                        comp_vec.push(component);
                    }
                }
            } else if let Some(compile_flags) = directive.strip_prefix(COMPILE_FLAGS_HEADER) {
                if let Some((_, v)) = compile_flags.split_once("--target") {
                    let v = v.trim_start_matches([' ', '=']);
                    let v = if v == "{{target}}" { Some((v, v)) } else { v.split_once("-") };
                    if let Some((arch, _)) = v {
                        let info = header_map.entry(revision).or_insert(RevisionInfo::default());
                        info.target_arch.replace(arch);
                    } else {
                        eprintln!("{file}: seems to have a malformed --target value");
                        *bad = true;
                    }
                }
            }
        });

        // Skip run-make tests as revisions are not supported.
        if entry.path().strip_prefix(tests_path).is_ok_and(|rest| rest.starts_with("run-make")) {
            return;
        }

        for (rev, RevisionInfo { target_arch, llvm_components }) in &header_map {
            let rev = rev.unwrap_or("[unspecified]");
            match (target_arch, llvm_components) {
                (None, None) => {}
                (Some(_), None) => {
                    eprintln!(
                        "{file}: revision {rev} should specify `{LLVM_COMPONENTS_HEADER}` as it has `--target` set"
                    );
                    *bad = true;
                }
                (None, Some(_)) => {
                    eprintln!(
                        "{file}: revision {rev} should not specify `{LLVM_COMPONENTS_HEADER}` as it doesn't need `--target`"
                    );
                    *bad = true;
                }
                (Some(_), Some(_)) => {
                    // FIXME: check specified components against the target architectures we
                    // gathered.
                }
            }
        }
    });
}
