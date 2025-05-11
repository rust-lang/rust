use crate::spec::Target;
use crate::spec::base::nto_qnx;

pub(crate) fn target() -> Target {
    let mut target = nto_qnx::aarch64();
    target.metadata.description = Some("ARM64 QNX Neutrino 8.0 RTOS".into());
    target.options.pre_link_args =
        nto_qnx::pre_link_args(nto_qnx::ApiVariant::Default, nto_qnx::Arch::Aarch64);
    target.options.env = "nto80".into();
    target
}
