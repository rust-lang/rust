mod config;
mod constraint;
mod intrinsic;
mod types;
mod xml_parser;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use intrinsic::X86IntrinsicType;
use xml_parser::get_xml_intrinsics;

pub struct X86ArchitectureTest {
    intrinsics: Vec<Intrinsic<X86IntrinsicType>>,
}

impl SupportedArchitectureTest for X86ArchitectureTest {
    type IntrinsicImpl = X86IntrinsicType;

    fn intrinsics(&self) -> &[Intrinsic<X86IntrinsicType>] {
        &self.intrinsics
    }

    const NOTICE: &str = config::NOTICE;

    const PLATFORM_C_HEADERS: &[&str] = &["immintrin.h"];

    const PLATFORM_RUST_DEFINITIONS: &str = config::PLATFORM_RUST_DEFINITIONS;
    const PLATFORM_RUST_CFGS: &str = config::PLATFORM_RUST_CFGS;

    fn arch_flags(&self) -> Vec<&str> {
        vec![
            "-mavx",
            "-mavx2",
            "-mavx512f",
            "-msse2",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512cd",
            "-mavx512fp16",
            "-msha",
            "-msha512",
            "-msm3",
            "-msm4",
            "-mavxvnni",
            "-mavxvnniint8",
            "-mavxneconvert",
            "-mavxifma",
            "-mavxvnniint16",
            "-mavx512bf16",
            "-mavx512bitalg",
            "-mavx512ifma",
            "-mavx512vbmi",
            "-mavx512vbmi2",
            "-mavx512vnni",
            "-mavx512vpopcntdq",
            "-mavx512vp2intersect",
            "-mbmi",
            "-mbmi2",
            "-mgfni",
            "-mvaes",
            "-mvpclmulqdq",
            "-mlzcnt",
        ]
    }

    fn create(cli_options: ProcessedCli) -> Self {
        let mut intrinsics =
            get_xml_intrinsics(&cli_options.filename).expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        intrinsics.dedup_by(|a, b| a.name == b.name);

        let sample_percentage: usize = cli_options.sample_percentage as usize;
        let sample_size = (intrinsics.len() * sample_percentage) / 100;

        let intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| i.arguments.args.len() > 0)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .take(sample_size)
            .collect::<Vec<_>>();

        Self { intrinsics }
    }
}
