use crate::spec::Target;

pub(crate) fn target() -> Target {
    let mut base = super::i686_unknown_linux_musl::target();
    base.rustc_abi = None; // overwrite the SSE2 ABI set by the base target
    base.cpu = "pentium".into();
    base.llvm_target = "i586-unknown-linux-musl".into();
    // FIXME(compiler-team#422): musl targets should be dynamically linked by default.
    base.crt_static_default = true;
    base
}
