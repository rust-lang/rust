use crate::spec::{
    Cc, LinkArgs, LinkerFlavor, Lld, RelroLevel, Target, TargetMetadata, TargetOptions, cvs,
};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        crt_static_respected: true,
        dynamic_linking: true,
        executables: true,
        families: cvs!["unix"],
        has_rpath: true,
        has_thread_local: false,
        linker: Some("qcc".into()),
        os: "nto".into(),
        position_independent_executables: true,
        static_position_independent_executables: true,
        relro_level: RelroLevel::Full,
        ..Default::default()
    }
}

pub(crate) fn meta() -> TargetMetadata {
    TargetMetadata { description: None, tier: Some(3), host_tools: Some(false), std: Some(true) }
}

pub(crate) fn aarch64() -> Target {
    Target {
        llvm_target: "aarch64-unknown-unknown".into(),
        metadata: meta(),
        pointer_width: 64,
        // from: https://llvm.org/docs/LangRef.html#data-layout
        // e         = little endian
        // m:e       = ELF mangling: Private symbols get a .L prefix
        // i8:8:32   = 8-bit-integer, minimum_alignment=8, preferred_alignment=32
        // i16:16:32 = 16-bit-integer, minimum_alignment=16, preferred_alignment=32
        // i64:64    = 64-bit-integer, minimum_alignment=64, preferred_alignment=64
        // i128:128  = 128-bit-integer, minimum_alignment=128, preferred_alignment=128
        // n32:64    = 32 and 64 are native integer widths; Elements of this set are considered to support most general arithmetic operations efficiently.
        // S128      = 128 bits are the natural alignment of the stack in bits.
        data_layout: "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32".into(),
        arch: "aarch64".into(),
        options: TargetOptions {
            features: "+v8a".into(),
            max_atomic_width: Some(128),
            ..opts()
        }
    }
}

pub(crate) fn x86_64() -> Target {
    Target {
        llvm_target: "x86_64-pc-unknown".into(),
        metadata: meta(),
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: TargetOptions {
            cpu: "x86-64".into(),
            plt_by_default: false,
            max_atomic_width: Some(64),
            vendor: "pc".into(),
            ..opts()
        },
    }
}

pub(crate) fn pre_link_args(api_var: ApiVariant, arch: Arch) -> LinkArgs {
    let (qcc_arg, arch_lib_dir) = match arch {
        Arch::Aarch64 => ("-Vgcc_ntoaarch64le_cxx", "aarch64le"),
        Arch::I586 => {
            ("-Vgcc_ntox86_cxx", "notSupportedByQnx_compiler/rustc_target/src/spec/base/nto_qnx.rs")
        }
        Arch::X86_64 => ("-Vgcc_ntox86_64_cxx", "x86_64"),
    };
    match api_var {
        ApiVariant::Default => {
            TargetOptions::link_args(LinkerFlavor::Gnu(Cc::Yes, Lld::No), &[qcc_arg])
        }
        ApiVariant::IoSock => TargetOptions::link_args(
            LinkerFlavor::Gnu(Cc::Yes, Lld::No),
            &[qcc_arg, get_iosock_param(arch_lib_dir)],
        ),
    }
}

pub(crate) enum ApiVariant {
    Default,
    IoSock,
}

pub(crate) enum Arch {
    Aarch64,
    I586,
    X86_64,
}

// When using `io-sock` on QNX, we must add a search path for the linker so
// that it prefers the io-sock version.
// The path depends on the host, i.e. we cannot hard-code it here, but have
// to determine it when the compiler runs.
// When using the QNX toolchain, the environment variable QNX_TARGET is always set.
// More information:
// https://www.qnx.com/developers/docs/7.1/index.html#com.qnx.doc.neutrino.io_sock/topic/migrate_app.html
fn get_iosock_param(arch_lib_dir: &str) -> &'static str {
    let target_dir = std::env::var("QNX_TARGET")
        .unwrap_or_else(|_| "QNX_TARGET_not_set_please_source_qnxsdp-env.sh".into());
    let linker_param = format!("-L{target_dir}/{arch_lib_dir}/io-sock/lib");

    // FIXME: leaking this is kind of weird: we're feeding these into something that expects an
    // `AsRef<OsStr>`, but often converts to `OsString` anyways, so shouldn't we just demand an `OsString`?
    linker_param.leak()
}
