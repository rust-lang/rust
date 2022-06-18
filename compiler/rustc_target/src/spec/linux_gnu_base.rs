use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions { env: "gnu".into(), ..super::linux_base::opts() }
}
