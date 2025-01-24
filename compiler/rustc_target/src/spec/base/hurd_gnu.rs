use crate::spec::{TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions { env: "gnu".into(), ..base::hurd::opts() }
}
