use crate::spec::{base, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions { env: "uclibc".into(), ..base::linux::opts() }
}
