//! Tidy check to ensure that all target specific tests (those that require a `--target` flag)
//! also require the pre-requisite LLVM components to run.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::walk::filter_not_rust;

const LLVM_COMPONENTS_HEADERS: &[&str] = &["needs-llvm-components:", "@needs-llvm-components:"];
const COMPILE_FLAGS_HEADERS: &[&str] = &["compile-flags:", "@compile-flags:"];

#[derive(Default, Debug)]
struct RevisionInfo {
    target_arch: Option<String>,
    llvm_components: Option<Vec<String>>,
}

pub fn check(path: &Path, bad: &mut bool) {
    crate::walk::walk(path, |path, _is_dir| filter_not_rust(path), &mut |entry, _| {
        let file = entry.path().display();
        let mut header_map = BTreeMap::new();
        let f = BufReader::new(File::open(entry.path()).unwrap());
        test_common::iter_header(entry.path(), f, &mut |comment| {
            // FIXME (ui_test) ideally this logic could be shared with compiletest
            // but compiletest needs to expand variables with its config
            if let Some(value) = LLVM_COMPONENTS_HEADERS
                .iter()
                .find_map(|prefix| comment.comment_str().strip_prefix(prefix))
            {
                let info = header_map
                    .entry(comment.revision().map(str::to_string))
                    .or_insert(RevisionInfo::default());
                let comp_vec = info.llvm_components.get_or_insert(Vec::new());
                for component in value.split(' ') {
                    let component = component.trim().to_string();
                    if !component.is_empty() {
                        comp_vec.push(component);
                    }
                }
            } else if let Some(compile_flags) = COMPILE_FLAGS_HEADERS
                .iter()
                .find_map(|prefix| comment.comment_str().strip_prefix(prefix))
            {
                if let Some((_, v)) = compile_flags.split_once("--target") {
                    if let Some((arch, _)) =
                        v.trim_start_matches(|c| c == ' ' || c == '=').split_once("-")
                    {
                        let info = header_map
                            .entry(comment.revision().map(str::to_string))
                            .or_insert(RevisionInfo::default());
                        info.target_arch.replace(arch.to_string());
                    } else {
                        eprintln!("{file}: seems to have a malformed --target value");
                        *bad = true;
                    }
                }
            }
        });
        for (rev, RevisionInfo { target_arch, llvm_components }) in header_map {
            let rev = rev.unwrap_or(String::from("[unspecified]"));
            match (target_arch, llvm_components) {
                (None, None) => {}
                (Some(_), None) => {
                    eprintln!(
                        "{}: revision {} should specify `@needs-llvm-components:` as it has `--target` set",
                        file, rev
                    );
                    *bad = true;
                }
                (None, Some(_)) => {
                    eprintln!(
                        "{}: revision {} should not specify `@needs-llvm-components:` as it doesn't need `--target`",
                        file, rev,
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
