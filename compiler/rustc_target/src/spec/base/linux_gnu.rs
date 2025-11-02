use crate::spec::{Cc, LinkerFlavor, Lld, TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    let mut base = TargetOptions { env: "gnu".into(), ..base::linux::opts() };

    // When we're asked to use the `rust-lld` linker by default, set the appropriate lld-using
    // linker flavor, and self-contained linker component.
    if option_env!("CFG_DEFAULT_LINKER_SELF_CONTAINED_LLD_CC").is_some() {
        base.linker_flavor = LinkerFlavor::Gnu(Cc::Yes, Lld::Yes);
        base.link_self_contained = crate::spec::LinkSelfContainedDefault::with_linker();
    }

    base
}
