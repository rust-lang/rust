use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions { env: "uclibc".to_string(), ..super::linux_base::opts() }
}
