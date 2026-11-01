mod intrinsic;
mod parser;
mod types;

use std::path::Path;

use crate::common::SupportedArchitecture;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use intrinsic::LoongArchType;
use parser::get_intrinsics;

#[derive(PartialEq)]
pub struct LoongArch {
    intrinsics: Vec<Intrinsic<LoongArch>>,
}

impl SupportedArchitecture for LoongArch {
    type Type = LoongArchType;

    fn intrinsics(&self) -> &[Intrinsic<Self>] {
        &self.intrinsics
    }

    const NOTICE: &str = r#"
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from LoongArch specification files, published under the same license as the
// `intrinsic-test` crate.
"#;

    const C_PRELUDE: &str = r#"
#include <lsxintrin.h>
#include <lasxintrin.h>
"#;
    const RUST_PRELUDE: &str = RUST_PRELUDE;

    const C_NAME_PREFIX: &str = "__";

    fn c_compiler_flags(&self, _cli_options: &ProcessedCli) -> Vec<&str> {
        let mut flags = vec!["-mlsx"];
        if self
            .intrinsics
            .iter()
            .any(|intrinsic| intrinsic.extension == "LASX")
        {
            flags.push("-mlasx");
        }
        if self.intrinsics.iter().any(|intrinsic| {
            intrinsic.name.contains("frecipe") || intrinsic.name.contains("frsqrte")
        }) {
            flags.push("-mfrecipe");
        }
        flags
    }

    fn create(cli_options: &ProcessedCli) -> Self {
        let mut intrinsics =
            load_intrinsics(&cli_options.filename).expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        intrinsics.dedup_by(|a, b| {
            a.name == b.name && a.results == b.results && a.arguments == b.arguments
        });

        let intrinsics = intrinsics
            .into_iter()
            // Skip intrinsics that don't return a value.
            .filter(|intrinsic| intrinsic.results.kind() != TypeKind::Void)
            .filter(|intrinsic| !intrinsic.arguments.args.is_empty())
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|intrinsic| !intrinsic.arguments.iter().any(|arg| arg.is_ptr()))
            // Skip intrinsics from `--skip`
            .filter(|intrinsic| !cli_options.skip.contains(&intrinsic.name))
            .collect::<Vec<_>>();

        let sample_percentage: usize = cli_options.sample_percentage as usize;
        let sample_size = (intrinsics.len() * sample_percentage) / 100;
        let intrinsics = intrinsics.into_iter().take(sample_size).collect();

        Self { intrinsics }
    }

    fn predicate_function(_: u32) -> String {
        unimplemented!("no scalable vectors on LoongArch")
    }
}

fn load_intrinsics(path: &Path) -> Result<Vec<Intrinsic<LoongArch>>, Box<dyn std::error::Error>> {
    if path.is_dir() {
        let mut intrinsics = Vec::new();
        for spec in ["lsx.spec", "lasx.spec"] {
            let spec_path = path.join(spec);
            if spec_path.exists() {
                intrinsics.extend(get_intrinsics(&spec_path)?);
            }
        }
        return Ok(intrinsics);
    }

    get_intrinsics(path)
}

const RUST_PRELUDE: &str = r#"
#![feature(stdarch_loongarch)]

use core_arch::arch::loongarch64::*;

#[inline]
unsafe fn lsx_vld_to_m128i(mem_addr: *const i8) -> m128i {
    lsx_vld::<0>(mem_addr)
}

#[inline]
unsafe fn lsx_vld_to_m128(mem_addr: *const i8) -> m128 {
    core::mem::transmute(lsx_vld::<0>(mem_addr))
}

#[inline]
unsafe fn lsx_vld_to_m128d(mem_addr: *const i8) -> m128d {
    core::mem::transmute(lsx_vld::<0>(mem_addr))
}

#[inline]
unsafe fn lasx_xvld_to_m256i(mem_addr: *const i8) -> m256i {
    lasx_xvld::<0>(mem_addr)
}

#[inline]
unsafe fn lasx_xvld_to_m256(mem_addr: *const i8) -> m256 {
    core::mem::transmute(lasx_xvld::<0>(mem_addr))
}

#[inline]
unsafe fn lasx_xvld_to_m256d(mem_addr: *const i8) -> m256d {
    core::mem::transmute(lasx_xvld::<0>(mem_addr))
}
"#;
