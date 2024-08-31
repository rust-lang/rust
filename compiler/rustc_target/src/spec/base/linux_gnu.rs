use crate::spec::{base, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions { env: "gnu".into(), ..base::linux::opts() }
}
