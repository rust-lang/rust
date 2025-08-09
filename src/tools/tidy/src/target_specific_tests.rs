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
    target_arch: Option<Option<&'a str>>,
    llvm_components: Option<Vec<&'a str>>,
}

pub fn check(tests_path: &Path, bad: &mut bool) {
    crate::walk::walk(tests_path, |path, _is_dir| filter_not_rust(path), &mut |entry, content| {
        if content.contains("// ignore-tidy-target-specific-tests") {
            return;
        }

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
            } else if let Some(compile_flags) = directive.strip_prefix(COMPILE_FLAGS_HEADER)
                && let Some((_, v)) = compile_flags.split_once("--target")
            {
                let v = v.trim_start_matches([' ', '=']);
                let info = header_map.entry(revision).or_insert(RevisionInfo::default());
                if v.starts_with("{{") {
                    info.target_arch.replace(None);
                } else if let Some((arch, _)) = v.split_once("-") {
                    info.target_arch.replace(Some(arch));
                } else {
                    eprintln!("{file}: seems to have a malformed --target value");
                    *bad = true;
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
                (Some(target_arch), None) => {
                    let llvm_component =
                        target_arch.map_or_else(|| "<arch>".to_string(), arch_to_llvm_component);
                    eprintln!(
                        "{file}: revision {rev} should specify `{LLVM_COMPONENTS_HEADER} {llvm_component}` as it has `--target` set"
                    );
                    *bad = true;
                }
                (None, Some(_)) => {
                    eprintln!(
                        "{file}: revision {rev} should not specify `{LLVM_COMPONENTS_HEADER}` as it doesn't need `--target`"
                    );
                    *bad = true;
                }
                (Some(target_arch), Some(llvm_components)) => {
                    if let Some(target_arch) = target_arch {
                        let llvm_component = arch_to_llvm_component(target_arch);
                        if !llvm_components.contains(&llvm_component.as_str()) {
                            eprintln!(
                                "{file}: revision {rev} should specify `{LLVM_COMPONENTS_HEADER} {llvm_component}` as it has `--target` set"
                            );
                            *bad = true;
                        }
                    }
                }
            }
        }
    });
}

fn arch_to_llvm_component(arch: &str) -> String {
    // NOTE: This is an *approximate* mapping of Rust's `--target` architecture to LLVM component
    // names. It is not intended to be an authoritative source, but rather a best-effort that's good
    // enough for the purpose of this tidy check.
    match arch {
        "amdgcn" => "amdgpu".into(),
        "aarch64_be" | "arm64_32" | "arm64e" | "arm64ec" => "aarch64".into(),
        "i386" | "i586" | "i686" | "x86" | "x86_64" | "x86_64h" => "x86".into(),
        "loongarch32" | "loongarch64" => "loongarch".into(),
        "nvptx64" => "nvptx".into(),
        "s390x" => "systemz".into(),
        "sparc64" | "sparcv9" => "sparc".into(),
        "wasm32" | "wasm32v1" | "wasm64" => "webassembly".into(),
        _ if arch.starts_with("armeb")
            || arch.starts_with("armv")
            || arch.starts_with("thumbv") =>
        {
            "arm".into()
        }
        _ if arch.starts_with("bpfe") => "bpf".into(),
        _ if arch.starts_with("mips") => "mips".into(),
        _ if arch.starts_with("powerpc") => "powerpc".into(),
        _ if arch.starts_with("riscv") => "riscv".into(),
        _ => arch.to_ascii_lowercase(),
    }
}
