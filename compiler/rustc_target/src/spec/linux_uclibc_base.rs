use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions { target_env: "uclibc".to_string(), ..super::linux_base::opts() }
}
