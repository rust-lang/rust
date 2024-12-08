use crate::spec::Target;

pub(crate) fn target() -> Target {
    let mut base = super::aarch64_unknown_nto_qnx700::target();
    base.metadata.description = Some("ARM64 QNX Neutrino 7.1 RTOS".into());
    base.options.env = "nto71".into();
    base
}
