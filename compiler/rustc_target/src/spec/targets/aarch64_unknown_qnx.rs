use crate::spec::base::qnx_sdp;
use crate::spec::{Os, Target};

pub(crate) fn target() -> Target {
    let mut target = qnx_sdp::aarch64();
    target.metadata.description = Some("ARM64 QNX SDP 8.0+".into());
    target.options.pre_link_args =
        qnx_sdp::pre_link_args(qnx_sdp::ApiVariant::Default, qnx_sdp::Arch::Aarch64);
    // for QNX SDP 8.0, we have target_os = "qnx" and no target_env
    target.options.os = Os::Qnx;
    target
}
