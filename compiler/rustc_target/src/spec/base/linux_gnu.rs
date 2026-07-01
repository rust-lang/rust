use crate::spec::{Cc, Env, LinkerFlavor, Lld, TargetOptions, base};

pub(crate) fn opts() -> TargetOptions {
    let mut base = TargetOptions { env: Env::Gnu, ..base::linux::opts() };

    // When we're asked to use the `rust-lld` linker by default, set the appropriate lld-using
    // linker flavor, and self-contained linker component.
    if option_env!("CFG_DEFAULT_LINKER_SELF_CONTAINED_LLD_CC").is_some() {
        base.linker_flavor = LinkerFlavor::Gnu(Cc::Yes, Lld::Yes);
        base.link_self_contained = crate::spec::LinkSelfContainedDefault::with_linker();
    }

    // glibc supports unloading dynamic libraries through dlclose. However,
    // unloading dynamic libraries is fundamentally incompatible with assumptions
    // made by `std` and much of the Rust ecosystem. Notably, the `'static`
    // lifetime is commonly expected to last until the termination of the abstract
    // machine, which is required for the soundness of e.g. thread-local destructors.
    // Hence, we disable unloading for all Rust objects by having the linker set
    // the DF_1_NODELETE flag. Note that this flag does not make `dlclose` fail,
    // it will just not have any effect.
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &["-Wl,-z,nodelete"]);
    base.add_pre_link_args(LinkerFlavor::Gnu(Cc::No, Lld::No), &["-z", "nodelete"]);

    base
}
