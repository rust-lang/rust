use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions { env: "gnu".into(), ..super::hurd_base::opts() }
}
