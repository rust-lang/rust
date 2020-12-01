use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions { env: "gnu".to_string(), ..super::linux_base::opts() }
}
