use crate::spec::{Env, TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions { env: Env::Gnu, ..base::hurd::opts() }
}
