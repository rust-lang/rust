use crate::spec::{Cc, LinkerFlavor, Lld, Target, TargetOptions};

pub(crate) fn target() -> Target {
    let mut target = super::x86_64_pc_nto_qnx710::target();
    target.options.env = "nto71_iosock".into();
    target.options.pre_link_args =
        TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &[
            "-Vgcc_ntox86_64_cxx",
            get_iosock_param(),
        ]);
    target
}

// When using `io-sock` on QNX, we must add a search path for the linker so
// that it prefers the io-sock version.
// The path depends on the host, i.e. we cannot hard-code it here, but have
// to determine it when the compiler runs.
// When using the QNX toolchain, the environment variable QNX_TARGET is always set.
// More information:
// https://www.qnx.com/developers/docs/7.1/index.html#com.qnx.doc.neutrino.io_sock/topic/migrate_app.html
fn get_iosock_param() -> &'static str {
    let target_dir =
        std::env::var("QNX_TARGET").unwrap_or_else(|_| "PLEASE_SET_ENV_VAR_QNX_TARGET".into());
    let linker_param = format!("-L{target_dir}/x86_64/io-sock/lib");

    linker_param.leak()
}
