//! Tests for target tier policy compliance.
//!
//! As of writing, only checks that sanity-check assembly test for targets doesn't miss any targets.

use std::collections::HashSet;
use std::path::Path;

use crate::walk::{filter_not_rust, walk};

const TARGET_DEFINITIONS_PATH: &str = "compiler/rustc_target/src/spec/targets/";
const ASSEMBLY_LLVM_TEST_PATH: &str = "tests/assembly-llvm/targets/";
const REVISION_LINE_START: &str = "//@ revisions: ";
const EXCEPTIONS: &[&str] = &[
    // FIXME: disabled since it fails on CI saying the csky component is missing
    "csky_unknown_linux_gnuabiv2",
    "csky_unknown_linux_gnuabiv2hf",
    // FIXME: disabled since it requires a custom LLVM until the upstream LLVM adds support for the target (https://github.com/espressif/llvm-project/issues/4)
    "xtensa_esp32_none_elf",
    "xtensa_esp32_espidf",
    "xtensa_esp32s2_none_elf",
    "xtensa_esp32s2_espidf",
    "xtensa_esp32s3_none_elf",
    "xtensa_esp32s3_espidf",
];

pub fn check(root_path: &Path, bad: &mut bool) {
    let mut targets_to_find = HashSet::new();

    let definitions_path = root_path.join(TARGET_DEFINITIONS_PATH);
    for defn in ignore::WalkBuilder::new(&definitions_path)
        .max_depth(Some(1))
        .filter_entry(|e| !filter_not_rust(e.path()))
        .build()
    {
        let defn = defn.unwrap();
        // Skip directory itself.
        if defn.path() == definitions_path {
            continue;
        }

        let path = defn.path();
        let target_name = path.file_stem().unwrap().to_string_lossy().into_owned();
        let _ = targets_to_find.insert(target_name);
    }

    walk(&root_path.join(ASSEMBLY_LLVM_TEST_PATH), |_, _| false, &mut |_, contents| {
        for line in contents.lines() {
            let Some(_) = line.find(REVISION_LINE_START) else {
                continue;
            };
            let (_, target_name) = line.split_at(REVISION_LINE_START.len());
            targets_to_find.remove(target_name);
        }
    });

    for target in targets_to_find {
        if !EXCEPTIONS.contains(&target.as_str()) {
            tidy_error!(bad, "{ASSEMBLY_LLVM_TEST_PATH}: missing assembly test for {target}")
        }
    }
}
