use crate::spec::{base, TargetOptions};

pub fn opts() -> TargetOptions {
    TargetOptions { env: "gnu".into(), ..base::hurd::opts() }
}
