use crate::spec::{base, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions { env: "uclibc".into(), ..base::linux::opts() }
}
