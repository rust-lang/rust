//! Tidy check to ensure that all target specific tests (those that require a `--target` flag)
//! also require the pre-requisite LLVM components to run.

use std::collections::BTreeMap;
use std::path::Path;

use crate::walk::filter_not_rust;

const COMMENT: &str = "//";
const LLVM_COMPONENTS_HEADER: &str = "needs-llvm-components:";
const COMPILE_FLAGS_HEADER: &str = "compile-flags:";

/// Iterate through compiletest headers in a test contents.
///
/// Adjusted from compiletest/src/header.rs.
fn iter_header<'a>(contents: &'a str, it: &mut dyn FnMut(Option<&'a str>, &'a str)) {
    for ln in contents.lines() {
        let ln = ln.trim();
        if ln.starts_with(COMMENT) && ln[COMMENT.len()..].trim_start().starts_with('[') {
            if let Some(close_brace) = ln.find(']') {
                let open_brace = ln.find('[').unwrap();
                let lncfg = &ln[open_brace + 1..close_brace];
                it(Some(lncfg), ln[(close_brace + 1)..].trim_start());
            } else {
                panic!("malformed condition directive: expected `//[foo]`, found `{ln}`")
            }
        } else if ln.starts_with(COMMENT) {
            it(None, ln[COMMENT.len()..].trim_start());
        }
    }
}

#[derive(Default, Debug)]
struct RevisionInfo<'a> {
    target_arch: Option<&'a str>,
    llvm_components: Option<Vec<&'a str>>,
}

pub fn check(path: &Path, bad: &mut bool) {
    crate::walk::walk(path, |path, _is_dir| filter_not_rust(path), &mut |entry, content| {
        let file = entry.path().display();
        let mut header_map = BTreeMap::new();
        iter_header(content, &mut |cfg, directive| {
            if let Some(value) = directive.strip_prefix(LLVM_COMPONENTS_HEADER) {
                let info = header_map.entry(cfg).or_insert(RevisionInfo::default());
                let comp_vec = info.llvm_components.get_or_insert(Vec::new());
                for component in value.split(' ') {
                    let component = component.trim();
                    if !component.is_empty() {
                        comp_vec.push(component);
                    }
                }
            } else if directive.starts_with(COMPILE_FLAGS_HEADER) {
                let compile_flags = &directive[COMPILE_FLAGS_HEADER.len()..];
                if let Some((_, v)) = compile_flags.split_once("--target") {
                    if let Some((arch, _)) =
                        v.trim_start_matches(|c| c == ' ' || c == '=').split_once("-")
                    {
                        let info = header_map.entry(cfg).or_insert(RevisionInfo::default());
                        info.target_arch.replace(arch);
                    } else {
                        eprintln!("{file}: seems to have a malformed --target value");
                        *bad = true;
                    }
                }
            }
        });
        for (rev, RevisionInfo { target_arch, llvm_components }) in &header_map {
            let rev = rev.unwrap_or("[unspecified]");
            match (target_arch, llvm_components) {
                (None, None) => {}
                (Some(_), None) => {
                    eprintln!(
                        "{}: revision {} should specify `{}` as it has `--target` set",
                        file, rev, LLVM_COMPONENTS_HEADER
                    );
                    *bad = true;
                }
                (None, Some(_)) => {
                    eprintln!(
                        "{}: revision {} should not specify `{}` as it doesn't need `--target`",
                        file, rev, LLVM_COMPONENTS_HEADER
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
