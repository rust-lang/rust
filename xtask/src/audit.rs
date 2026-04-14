use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::Context;
use cargo_metadata::MetadataCommand;

use crate::common::Result;

const ALLOWED_STD_CRATES: &[&str] = &[
    "xtask",
    "pciids",
    "bdd",
    "unifont-gen",
    "display_proto_tests",
    "abi-macros",
    "stem-macros",
];

const REQUIRED_NOSTD_CRATES: &[&str] =
    &["kernel", "stem", "stem-macros", "abi", "abi-macros", "bran"];

pub fn audit() -> Result<()> {
    println!("Platform Boundary Audit");
    println!("============================================================");

    let metadata = MetadataCommand::new()
        .exec()
        .context("failed to run cargo metadata")?;

    let mut errors = Vec::new();
    let mut count = 0;

    let allowed_std: HashSet<&str> = ALLOWED_STD_CRATES.iter().copied().collect();
    let required_nostd: HashSet<&str> = REQUIRED_NOSTD_CRATES.iter().copied().collect();

    for package in &metadata.packages {
        if !metadata.workspace_members.contains(&package.id) {
            continue;
        }

        let name = package.name.as_str();

        if allowed_std.contains(name) {
            println!("ok   {:30} [std allowed - build tool]", name);
            continue;
        }

        let manifest_path = package.manifest_path.as_std_path();
        let is_userspace = manifest_path
            .components()
            .any(|component| component.as_os_str() == "userspace");
        let is_kernel_or_core = required_nostd.contains(name);

        if is_userspace || is_kernel_or_core {
            count += 1;
            let crate_root = manifest_path.parent().unwrap();
            if is_nostd_crate(crate_root) {
                println!("ok   {:30} [no_std compliant]", name);
            } else {
                println!("fail {:30} [missing #![no_std]]", name);
                errors.push(format!("{name} is missing #![no_std] declaration"));
            }
        }
    }

    println!("============================================================");
    println!("Checked {count} crates for no_std compliance");

    if !errors.is_empty() {
        println!("\nErrors:");
        for error in &errors {
            println!("  - {error}");
        }
        anyhow::bail!("audit failed with {} errors", errors.len());
    }

    println!("\nPlatform boundary audit passed.");
    Ok(())
}

fn is_nostd_crate(crate_path: &Path) -> bool {
    let src_dir = crate_path.join("src");
    for file_name in ["lib.rs", "main.rs"] {
        let file_path = src_dir.join(file_name);
        if !file_path.exists() {
            continue;
        }

        if let Ok(content) = fs::read_to_string(&file_path) {
            for line in content.lines().take(120) {
                if line.contains("#![no_std]")
                    || line.contains("#![cfg_attr(not(test), no_std)]")
                {
                    return true;
                }
            }
        }
    }

    false
}
