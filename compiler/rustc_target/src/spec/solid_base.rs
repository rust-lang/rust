use crate::spec::TargetOptions;

pub fn opts(kernel: &str) -> TargetOptions {
    TargetOptions {
        os: format!("solid-{}", kernel),
        vendor: "kmc".to_string(),
        has_elf_tls: true,
        ..Default::default()
    }
}
