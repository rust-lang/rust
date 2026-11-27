use crate::spec::{Env, TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions { env: Env::Uclibc, ..base::linux::opts() }
}
