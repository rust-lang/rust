use crate::spec::base::qnx_sdp;
use crate::spec::{Env, Target};

pub(crate) fn target() -> Target {
    let mut target = qnx_sdp::aarch64();
    target.metadata.description = Some("ARM64 QNX SDP 7.1 with io-pkt network stack".into());
    target.options.pre_link_args =
        qnx_sdp::pre_link_args(qnx_sdp::ApiVariant::Default, qnx_sdp::Arch::Aarch64);
    // for QNX SDP 7.x, we keep target_os = "nto" for backwards compatibility, and use target_env to specify which version
    target.options.env = Env::Nto71;
    target
}
