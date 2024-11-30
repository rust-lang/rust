use crate::spec::Target;
use crate::spec::base::nto_qnx;

pub(crate) fn target() -> Target {
    let mut target = nto_qnx::x86_64();
    target.metadata.description =
        Some("x86 64-bit QNX Neutrino 7.1 RTOS with io-sock network stack".into());
    target.options.pre_link_args =
        nto_qnx::pre_link_args(nto_qnx::ApiVariant::IoSock, nto_qnx::Arch::X86_64);
    target.options.env = "nto71_iosock".into();
    target
}
