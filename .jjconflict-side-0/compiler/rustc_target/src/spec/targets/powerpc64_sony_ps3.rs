use crate::spec::{
    Arch, Cc, CfgAbi, CodeModel, Endian, FramePointer, LinkerFlavor, Lld, LlvmAbi, Os,
    PanicStrategy, RelocModel, Target, TargetMetadata, TargetOptions,
};

pub(crate) fn target() -> Target {
    let pre_link_args = TargetOptions::link_args(
        LinkerFlavor::Gnu(Cc::No, Lld::No),
        &[
            // We strictly need ELFv1 PPC64.
            "-m",
            "elf64ppc",
            // PS3 LV2 reserves the first 64KB page for unmapped memory protection.
            "--image-base=0x10000",
            // Should be default, but relying on automatic behavior appears to be brittle.
            "-e",
            "_start",
            // CellOS expects .rodata to be merged into the executable Text segment (RX)
            //  so there are only 2 loadable segments (RX and RW)
            "--no-rosegment",
            // CellOS uses 64 KB memory pages. Without this flag, `mold` might align data segments to 4 KB boundaries.
            "-z",
            "separate-loadable-segments",
            // Prevents mold from creating a `PT_GNU_RELRO` segment that GameOS does not support.
            "-z",
            "norelro",
            // CellOS's loader doesn't behave like `ld`. PRXs are stubbed in the binary already.
            "-Bstatic",
            // The following are segments that might never be referenced by the code,
            //  but are expected to exist by the PS3's loader.
            "-u",
            "sys_process_param",
            "-u",
            "sys_proc_prx_param",
            "--undefined-glob=*_prx_header",
            "--undefined-glob=*_fnid_table",
            "--undefined-glob=*_name",
            "--undefined-glob=*_fstub_table",
        ],
    );

    Target {
        // LLVM will default to a compatible ELF backend.
        llvm_target: "powerpc64-sony-ps3".into(),

        metadata: TargetMetadata {
            description: Some("PowerPC64 (big endian) Sony PlayStation 3 (PS3)".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },

        // We declare pointers to be 64-bit as the PPU _is_ a 64-bit core.
        // However, for all real usage the OS limits us to **32-bit pointers**.
        // SDKs should therefore take this into account, specifically when handling syscalls.
        pointer_width: 64,

        data_layout: "E-m:e-Fi64-i64:64-i128:128-n32:64".into(),
        arch: Arch::PowerPC64,

        options: TargetOptions {
            // Base PS3 hardware.
            vendor: "sony".into(),
            endian: Endian::Big,
            os: Os::Ps3,
            cfg_abi: CfgAbi::ElfV1,
            llvm_abiname: LlvmAbi::ElfV1,
            features: "+altivec".into(),

            // CellOS requiring ELFv1 makes LLVM's `lld` incompatible.
            // See:
            // - [rust-lang/rust#85589](https://github.com/rust-lang/rust/issues/85589)
            // - [llvm/llvm-project#27630](https://github.com/llvm/llvm-project/issues/27630)
            linker: Some("mold".into()),
            linker_flavor: LinkerFlavor::Gnu(Cc::No, Lld::No),

            // CellOS _is_ case-sensitive, but the PS3's binaries vary
            //  in casing depending on whether they are games in `/dev_hdd0`
            //  or system binaries (such as PRX files).
            //
            // All games use the .ELF (uppercase) suffix, and Sony's own
            //  documentation and tools expect user app binaries to be uppercase.
            exe_suffix: ".ELF".into(),

            // This limits us to 64KB of ToC, but yields smaller binaries and less assembly.
            // Only becomes a problem for binaries with thousands of dependencies.
            code_model: Some(CodeModel::Small),
            // Prevents LLVM from emitting modern linker relaxation relocations.
            relax_elf_relocations: false,
            // CellOS main executables (`EBOOT.ELF`) **must be static executables** (ET_EXEC).
            relocation_model: RelocModel::Static,

            // Locking defaults against future changes.
            c_int_width: 32,
            executables: true,
            frame_pointer: FramePointer::MayOmit,
            // Change this to `true` for developing kernel-mode applications.
            // This target defaults to user-mode, and the kernel already handles
            //  this for us, so keeping it off is a performance gain.
            disable_redzone: false,

            panic_strategy: PanicStrategy::Abort,
            pre_link_args,
            ..Default::default()
        },
    }
}
