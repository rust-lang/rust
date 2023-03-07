use crate::spec::TargetOptions;

pub fn opts() -> TargetOptions {
    TargetOptions { env: "uclibc".into(), ..super::linux_base::opts() }
}
