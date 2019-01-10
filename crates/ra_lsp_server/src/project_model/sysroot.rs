use std::{
    path::{Path, PathBuf},
    process::Command,
};

use ra_syntax::SmolStr;
use rustc_hash::FxHashMap;

use crate::Result;

#[derive(Debug, Clone)]
pub struct Sysroot {
    crates: FxHashMap<SmolStr, PathBuf>,
}

impl Sysroot {
    pub(crate) fn discover(cargo_toml: &Path) -> Result<Sysroot> {
        let rustc_output = Command::new("rustc")
            .current_dir(cargo_toml.parent().unwrap())
            .args(&["--print", "sysroot"])
            .output()?;
        if !rustc_output.status.success() {
            failure::bail!("failed to locate sysroot")
        }
        let stdout = String::from_utf8(rustc_output.stdout)?;
        let sysroot_path = Path::new(stdout.trim());
        let src = sysroot_path.join("lib/rustlib/src/rust/src");

        let crates: &[(&str, &[&str])] = &[
            (
                "std",
                &[
                    "alloc_jemalloc",
                    "alloc_system",
                    "panic_abort",
                    "rand",
                    "compiler_builtins",
                    "unwind",
                    "rustc_asan",
                    "rustc_lsan",
                    "rustc_msan",
                    "rustc_tsan",
                    "build_helper",
                ],
            ),
            ("core", &[]),
            ("alloc", &[]),
            ("collections", &[]),
            ("libc", &[]),
            ("panic_unwind", &[]),
            ("proc_macro", &[]),
            ("rustc_unicode", &[]),
            ("std_unicode", &[]),
            ("test", &[]),
            // Feature gated
            ("alloc_jemalloc", &[]),
            ("alloc_system", &[]),
            ("compiler_builtins", &[]),
            ("getopts", &[]),
            ("panic_unwind", &[]),
            ("panic_abort", &[]),
            ("rand", &[]),
            ("term", &[]),
            ("unwind", &[]),
            // Dependencies
            ("build_helper", &[]),
            ("rustc_asan", &[]),
            ("rustc_lsan", &[]),
            ("rustc_msan", &[]),
            ("rustc_tsan", &[]),
            ("syntax", &[]),
        ];

        Ok(Sysroot {
            crates: FxHashMap::default(),
        })
    }
}
