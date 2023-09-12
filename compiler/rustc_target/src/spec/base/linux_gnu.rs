use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions { env: "gnu".into(), ..super::linux::opts() }
}
