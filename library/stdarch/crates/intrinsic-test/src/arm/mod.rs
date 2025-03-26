mod argument;
mod format;
mod functions;
mod intrinsic;
mod json_parser;
mod types;

use crate::common::cli::ProcessedCli;
use crate::common::supporting_test::SupportedArchitectureTest;
use functions::{build_c, build_rust, compare_outputs};
use intrinsic::Intrinsic;
use json_parser::get_neon_intrinsics;
use types::TypeKind;

fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from a JSON specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}

pub struct ArmTestProcessor {
    intrinsics: Vec<Intrinsic>,
    notices: String,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for ArmTestProcessor {
    fn create(cli_options: ProcessedCli) -> Self {
        let a32 = cli_options.target.contains("v7");
        let mut intrinsics =
            get_neon_intrinsics(&cli_options.filename).expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));

        let mut intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .filter(|i| !(a32 && i.a64_only))
            .collect::<Vec<_>>();
        intrinsics.dedup();

        let notices = build_notices("// ");

        Self {
            intrinsics: intrinsics,
            notices: notices,
            cli_options: cli_options,
        }
    }

    fn build_c_file(&self) -> bool {
        build_c(
            &self.notices,
            &self.intrinsics,
            self.cli_options.cpp_compiler.as_deref(),
            &self.cli_options.target,
            self.cli_options.cxx_toolchain_dir.as_deref(),
        )
    }

    fn build_rust_file(&self) -> bool {
        build_rust(
            &self.notices,
            &self.intrinsics,
            self.cli_options.toolchain.as_deref(),
            &self.cli_options.target,
            self.cli_options.linker.as_deref(),
        )
    }

    fn compare_outputs(&self) -> bool {
        if let Some(ref toolchain) = self.cli_options.toolchain {
            compare_outputs(
                &self.intrinsics,
                toolchain,
                &self.cli_options.c_runner,
                &self.cli_options.target,
            )
        } else {
            true
        }
    }
}
