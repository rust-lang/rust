//! Clean tasks.

use crate::common::Result;
use xshell::{Shell, cmd};

/// Clean build artifacts plus fetched/vendor state.
pub fn clean(sh: &Shell) -> Result<()> {
    println!("Cleaning build artifacts and fetched state...");
    cmd!(sh, "cargo clean").run()?;
    sh.remove_path("iso_root")?;

    for entry in std::fs::read_dir(".").into_iter().flatten() {
        if let Ok(entry) = entry {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("thing-os-")
                && (name_str.ends_with(".iso") || name_str.ends_with(".hdd"))
            {
                println!("Removing {}", name_str);
                let _ = sh.remove_path(entry.path());
            }
        }
    }

    sh.remove_path("bran/bin-x86_64")?;
    sh.remove_path("bran/bin-aarch64")?;
    sh.remove_path("bran/bin-riscv64")?;
    sh.remove_path("bran/bin-loongarch64")?;

    println!("Cleaning fetched vendor trees...");
    sh.remove_path("vendor/limine")?;
    sh.remove_path("vendor/ovmf")?;
    sh.remove_path("vendor/future-cursors")?;

    println!("Cleaning downloaded assets (preserving wallpapers)...");
    sh.remove_path("assets/cursors")?;
    sh.remove_path("assets/fonts")?;

    if std::path::Path::new("assets/icons").exists() {
        for entry in std::fs::read_dir("assets/icons")? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name() {
                if name != "thingos" {
                    if path.is_dir() {
                        sh.remove_path(path)?;
                    } else {
                        std::fs::remove_file(path)?;
                    }
                }
            }
        }
    }
    sh.remove_path("assets/pci")?;

    Ok(())
}

/// Compatibility alias for `clean`.
pub fn distclean(sh: &Shell) -> Result<()> {
    clean(sh)
}
