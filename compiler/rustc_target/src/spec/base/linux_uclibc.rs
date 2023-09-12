use crate::spec::{base::linux, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions { env: "uclibc".into(), ..linux::opts() }
}
