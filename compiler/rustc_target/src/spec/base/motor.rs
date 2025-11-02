use crate::spec::{
    Cc, FramePointer, LinkerFlavor, Lld, PanicStrategy, StackProbeType, TargetOptions,
};

pub(crate) fn opts() -> TargetOptions {
    let pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &[
            "-e",
            "motor_start",
            "--no-undefined",
            "--error-unresolved-symbols",
            "--no-undefined-version",
            "-u",
            "__rust_abort",
        ],
    );
    TargetOptions {
        os: "motor".into(),
        executables: true,
        // TLS is false below because if true, the compiler assumes
        // we handle TLS at the ELF loading level, which we don't.
        // We use "OS level" TLS (see thread/local.rs in stdlib).
        has_thread_local: false,
        frame_pointer: FramePointer::NonLeaf,
        linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::No),
        main_needs_argc_argv: true,
        panic_strategy: PanicStrategy::Abort,
        pre_link_args,
        stack_probes: StackProbeType::Inline,
        supports_stack_protector: true,
        ..Default::default()
    }
}
