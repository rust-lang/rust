use crate::spec::{TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions { env: "uclibc".into(), ..base::linux::opts() }
}
