use core::result::Result;

use rustc_abi::Endian;
use rustc_data_structures::fx::FxHashSet;

use crate::spec::{
    Arch, Cc, CfgAbi, Env, FloatAbi, LinkerFlavor, Lld, LlvmAbi, Os, RelocModel, RustcAbi, Target,
    TargetKind,
};

impl Target {
    /// Check some basic consistency of the current target. For JSON targets we are less strict;
    /// some of these checks are more guidelines than strict rules.
    pub(super) fn check_consistency(&self, kind: TargetKind) -> Result<(), String> {
        macro_rules! check {
            ($b:expr, $($msg:tt)*) => {
                if !$b {
                    return Err(format!($($msg)*));
                }
            }
        }
        macro_rules! check_eq {
            ($left:expr, $right:expr, $($msg:tt)*) => {
                if ($left) != ($right) {
                    return Err(format!($($msg)*));
                }
            }
        }
        macro_rules! check_ne {
            ($left:expr, $right:expr, $($msg:tt)*) => {
                if ($left) == ($right) {
                    return Err(format!($($msg)*));
                }
            }
        }
        macro_rules! check_matches {
            ($left:expr, $right:pat, $($msg:tt)*) => {
                if !matches!($left, $right) {
                    return Err(format!($($msg)*));
                }
            }
        }

        check_eq!(
            self.is_like_darwin,
            self.vendor == "apple",
            "`is_like_darwin` must be set if and only if `vendor` is `apple`"
        );
        check_eq!(
            self.is_like_solaris,
            matches!(self.os, Os::Solaris | Os::Illumos),
            "`is_like_solaris` must be set if and only if `os` is `solaris` or `illumos`"
        );
        check_eq!(
            self.is_like_gpu,
            self.arch == Arch::Nvptx64 || self.arch == Arch::AmdGpu,
            "`is_like_gpu` must be set if and only if `target` is `nvptx64` or `amdgcn`"
        );
        check_eq!(
            self.is_like_windows,
            matches!(self.os, Os::Windows | Os::Uefi | Os::Cygwin),
            "`is_like_windows` must be set if and only if `os` is `windows`, `uefi` or `cygwin`"
        );
        check_eq!(
            self.is_like_wasm,
            matches!(self.arch, Arch::Wasm32 | Arch::Wasm64),
            "`is_like_wasm` must be set if and only if `arch` is `wasm32` or `wasm64`"
        );
        if self.is_like_msvc {
            check!(self.is_like_windows, "if `is_like_msvc` is set, `is_like_windows` must be set");
        }
        if self.os == Os::Emscripten {
            check!(self.is_like_wasm, "the `emcscripten` os only makes sense on wasm-like targets");
        }

        // Check that default linker flavor is compatible with some other key properties.
        check_eq!(
            self.is_like_darwin,
            matches!(self.linker_flavor, LinkerFlavor::Darwin(..)),
            "`linker_flavor` must be `darwin` if and only if `is_like_darwin` is set"
        );
        check_eq!(
            self.is_like_msvc,
            matches!(self.linker_flavor, LinkerFlavor::Msvc(..)),
            "`linker_flavor` must be `msvc` if and only if `is_like_msvc` is set"
        );
        check_eq!(
            self.is_like_wasm && self.os != Os::Emscripten,
            matches!(self.linker_flavor, LinkerFlavor::WasmLld(..)),
            "`linker_flavor` must be `wasm-lld` if and only if `is_like_wasm` is set and the `os` is not `emscripten`",
        );
        check_eq!(
            self.os == Os::Emscripten,
            matches!(self.linker_flavor, LinkerFlavor::EmCc),
            "`linker_flavor` must be `em-cc` if and only if `os` is `emscripten`"
        );
        check_eq!(
            self.arch == Arch::Bpf,
            matches!(self.linker_flavor, LinkerFlavor::Bpf),
            "`linker_flavor` must be `bpf` if and only if `arch` is `bpf`"
        );

        for args in [
            &self.pre_link_args,
            &self.late_link_args,
            &self.late_link_args_dynamic,
            &self.late_link_args_static,
            &self.post_link_args,
        ] {
            for (&flavor, flavor_args) in args {
                check!(
                    !flavor_args.is_empty() || self.arch == Arch::Avr,
                    "linker flavor args must not be empty"
                );
                // Check that flavors mentioned in link args are compatible with the default flavor.
                match self.linker_flavor {
                    LinkerFlavor::Gnu(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Gnu(..),
                            "mixing GNU and non-GNU linker flavors"
                        );
                    }
                    LinkerFlavor::Darwin(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Darwin(..),
                            "mixing Darwin and non-Darwin linker flavors"
                        )
                    }
                    LinkerFlavor::WasmLld(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::WasmLld(..),
                            "mixing wasm and non-wasm linker flavors"
                        )
                    }
                    LinkerFlavor::Unix(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Unix(..),
                            "mixing unix and non-unix linker flavors"
                        );
                    }
                    LinkerFlavor::Msvc(..) => {
                        check_matches!(
                            flavor,
                            LinkerFlavor::Msvc(..),
                            "mixing MSVC and non-MSVC linker flavors"
                        );
                    }
                    LinkerFlavor::EmCc | LinkerFlavor::Bpf | LinkerFlavor::Llbc => {
                        check_eq!(flavor, self.linker_flavor, "mixing different linker flavors")
                    }
                }

                // Check that link args for cc and non-cc versions of flavors are consistent.
                let check_noncc = |noncc_flavor| -> Result<(), String> {
                    if let Some(noncc_args) = args.get(&noncc_flavor) {
                        for arg in flavor_args {
                            if let Some(suffix) = arg.strip_prefix("-Wl,") {
                                check!(
                                    noncc_args.iter().any(|a| a == suffix),
                                    " link args for cc and non-cc versions of flavors are not consistent"
                                );
                            }
                        }
                    }
                    Ok(())
                };

                match self.linker_flavor {
                    LinkerFlavor::Gnu(Cc::Yes, lld) => check_noncc(LinkerFlavor::Gnu(Cc::No, lld))?,
                    LinkerFlavor::WasmLld(Cc::Yes) => check_noncc(LinkerFlavor::WasmLld(Cc::No))?,
                    LinkerFlavor::Unix(Cc::Yes) => check_noncc(LinkerFlavor::Unix(Cc::No))?,
                    _ => {}
                }
            }

            // Check that link args for lld and non-lld versions of flavors are consistent.
            for cc in [Cc::No, Cc::Yes] {
                check_eq!(
                    args.get(&LinkerFlavor::Gnu(cc, Lld::No)),
                    args.get(&LinkerFlavor::Gnu(cc, Lld::Yes)),
                    "link args for lld and non-lld versions of flavors are not consistent",
                );
                check_eq!(
                    args.get(&LinkerFlavor::Darwin(cc, Lld::No)),
                    args.get(&LinkerFlavor::Darwin(cc, Lld::Yes)),
                    "link args for lld and non-lld versions of flavors are not consistent",
                );
            }
            check_eq!(
                args.get(&LinkerFlavor::Msvc(Lld::No)),
                args.get(&LinkerFlavor::Msvc(Lld::Yes)),
                "link args for lld and non-lld versions of flavors are not consistent",
            );
        }

        if self.link_self_contained.is_disabled() {
            check!(
                self.pre_link_objects_self_contained.is_empty()
                    && self.post_link_objects_self_contained.is_empty(),
                "if `link_self_contained` is disabled, then `pre_link_objects_self_contained` and `post_link_objects_self_contained` must be empty",
            );
        }

        // If your target really needs to deviate from the rules below,
        // except it and document the reasons.
        // Keep the default "unknown" vendor instead.
        check_ne!(self.vendor, "", "`vendor` cannot be empty");
        if let Os::Other(s) = &self.os {
            check!(!s.is_empty(), "`os` cannot be empty");
        }
        if !self.can_use_os_unknown() {
            // Keep the default "none" for bare metal targets instead.
            check_ne!(
                self.os,
                Os::Unknown,
                "`unknown` os can only be used on particular targets; use `none` for bare-metal targets"
            );
        }

        // Check dynamic linking stuff.
        // We skip this for JSON targets since otherwise, our default values would fail this test.
        // These checks are not critical for correctness, but more like default guidelines.
        // FIXME (https://github.com/rust-lang/rust/issues/133459): do we want to change the JSON
        // target defaults so that they pass these checks?
        if kind == TargetKind::Builtin {
            // BPF: when targeting user space vms (like rbpf), those can load dynamic libraries.
            // hexagon: when targeting QuRT, that OS can load dynamic libraries.
            // wasm{32,64}: dynamic linking is inherent in the definition of the VM.
            if self.os == Os::None
                && !matches!(self.arch, Arch::Bpf | Arch::Hexagon | Arch::Wasm32 | Arch::Wasm64)
            {
                check!(
                    !self.dynamic_linking,
                    "dynamic linking is not supported on this OS/architecture"
                );
            }
            if self.only_cdylib
                || self.crt_static_allows_dylibs
                || !self.late_link_args_dynamic.is_empty()
            {
                check!(
                    self.dynamic_linking,
                    "dynamic linking must be allowed when `only_cdylib` or `crt_static_allows_dylibs` or `late_link_args_dynamic` are set"
                );
            }
            // Apparently PIC was slow on wasm at some point, see comments in wasm_base.rs
            if self.dynamic_linking && !self.is_like_wasm {
                check_eq!(
                    self.relocation_model,
                    RelocModel::Pic,
                    "targets that support dynamic linking must use the `pic` relocation model"
                );
            }
            if self.position_independent_executables {
                check_eq!(
                    self.relocation_model,
                    RelocModel::Pic,
                    "targets that support position-independent executables must use the `pic` relocation model"
                );
            }
            // The UEFI targets do not support dynamic linking but still require PIC (#101377).
            if self.relocation_model == RelocModel::Pic && self.os != Os::Uefi {
                check!(
                    self.dynamic_linking || self.position_independent_executables,
                    "when the relocation model is `pic`, the target must support dynamic linking or use position-independent executables. \
                Set the relocation model to `static` to avoid this requirement"
                );
            }
            if self.static_position_independent_executables {
                check!(
                    self.position_independent_executables,
                    "if `static_position_independent_executables` is set, then `position_independent_executables` must be set"
                );
            }
            if self.position_independent_executables {
                check!(
                    self.executables,
                    "if `position_independent_executables` is set then `executables` must be set"
                );
            }
        }

        // Check crt static stuff
        if self.crt_static_default || self.crt_static_allows_dylibs {
            check!(
                self.crt_static_respected,
                "static CRT can be enabled but `crt_static_respected` is not set"
            );
        }

        // Ensure built-in targets don't use the `Other` variants.
        if kind == TargetKind::Builtin {
            check!(
                !matches!(self.arch, Arch::Other(_)),
                "`Arch::Other` is only meant for JSON targets"
            );
            check!(!matches!(self.os, Os::Other(_)), "`Os::Other` is only meant for JSON targets");
            check!(
                !matches!(self.env, Env::Other(_)),
                "`Env::Other` is only meant for JSON targets"
            );
            check!(
                !matches!(self.cfg_abi, CfgAbi::Other(_)),
                "`CfgAbi::Other` is only meant for JSON targets"
            );
            check!(
                !matches!(self.llvm_abiname, LlvmAbi::Other(_)),
                "`LlvmAbi::Other` is only meant for JSON targets"
            );
        }

        // Check ABI flag consistency, for the architectures where we have proper ABI treatment.
        // To ensure targets are trated consistently, please consult with the team before allowing
        // new cases.
        match self.arch {
            Arch::X86 => {
                check!(
                    self.llvm_abiname == LlvmAbi::Unspecified,
                    "`llvm_abiname` is unused on x86-32"
                );
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on x86-32");
                check_matches!(
                    (&self.rustc_abi, &self.cfg_abi),
                    // FIXME: we do not currently set a target_abi for softfloat targets here,
                    // but we probably should, so we already allow it.
                    (
                        Some(RustcAbi::Softfloat),
                        CfgAbi::SoftFloat | CfgAbi::Unspecified | CfgAbi::Other(_)
                    ) | (
                        Some(RustcAbi::X86Sse2) | None,
                        CfgAbi::Uwp
                            | CfgAbi::Llvm
                            | CfgAbi::Sim
                            | CfgAbi::Unspecified
                            | CfgAbi::Other(_)
                    ),
                    "invalid x86-32 Rust-specific ABI and `cfg(target_abi)` combination:\n\
                    Rust-specific ABI: {:?}\n\
                    cfg(target_abi): {}",
                    self.rustc_abi,
                    self.cfg_abi,
                );
            }
            Arch::X86_64 => {
                check!(
                    self.llvm_abiname == LlvmAbi::Unspecified,
                    "`llvm_abiname` is unused on x86-64"
                );
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on x86-64");
                // FIXME: we do not currently set a target_abi for softfloat targets here, but we
                // probably should, so we already allow it.
                // FIXME: Ensure that target_abi = "x32" correlates with actually using that ABI.
                // Do any of the others need a similar check?
                check_matches!(
                    (&self.rustc_abi, &self.cfg_abi),
                    (
                        Some(RustcAbi::Softfloat),
                        CfgAbi::SoftFloat | CfgAbi::Unspecified | CfgAbi::Other(_)
                    ) | (
                        None,
                        CfgAbi::X32
                            | CfgAbi::Llvm
                            | CfgAbi::Fortanix
                            | CfgAbi::Uwp
                            | CfgAbi::MacAbi
                            | CfgAbi::Sim
                            | CfgAbi::Unspecified
                            | CfgAbi::Other(_)
                    ),
                    "invalid x86-64 Rust-specific ABI and `cfg(target_abi)` combination:\n\
                    Rust-specific ABI: {:?}\n\
                    cfg(target_abi): {}",
                    self.rustc_abi,
                    self.cfg_abi,
                );
            }
            Arch::RiscV32 => {
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on RISC-V");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on RISC-V");
                check_matches!(
                    (&self.llvm_abiname, &self.cfg_abi),
                    (LlvmAbi::Ilp32, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Ilp32f, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Ilp32d, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Ilp32e, CfgAbi::Ilp32e),
                    "invalid RISC-V ABI name and `cfg(target_abi)` combination:\n\
                     ABI name: {}\n\
                     cfg(target_abi): {}",
                    self.llvm_abiname,
                    self.cfg_abi,
                );
            }
            Arch::RiscV64 => {
                // Note that the `lp64e` is still unstable as it's not (yet) part of the ELF psABI.
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on RISC-V");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on RISC-V");
                check_matches!(
                    (&self.llvm_abiname, &self.cfg_abi),
                    (LlvmAbi::Lp64, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Lp64f, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Lp64d, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Lp64e, CfgAbi::Unspecified | CfgAbi::Other(_)),
                    "invalid RISC-V ABI name and `cfg(target_abi)` combination:\n\
                     ABI name: {}\n\
                     cfg(target_abi): {}",
                    self.llvm_abiname,
                    self.cfg_abi,
                );
            }
            Arch::Arm => {
                check!(
                    self.llvm_abiname == LlvmAbi::Unspecified,
                    "`llvm_abiname` is unused on ARM"
                );
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on ARM");
                check_matches!(
                    (&self.llvm_floatabi, &self.cfg_abi),
                    (
                        Some(FloatAbi::Hard),
                        CfgAbi::EabiHf | CfgAbi::Uwp | CfgAbi::Unspecified | CfgAbi::Other(_)
                    ) | (Some(FloatAbi::Soft), CfgAbi::Eabi),
                    "Invalid combination of float ABI and `cfg(target_abi)` for ARM target\n\
                     float ABI: {:?}\n\
                     cfg(target_abi): {}",
                    self.llvm_floatabi,
                    self.cfg_abi,
                )
            }
            Arch::AArch64 => {
                check_matches!(
                    self.llvm_abiname,
                    LlvmAbi::Unspecified | LlvmAbi::Pauthtest,
                    "invalid llvm ABI for aarch64"
                );
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on aarch64");
                // FIXME: Ensure that target_abi = "ilp32" correlates with actually using that ABI.
                // Do any of the others need a similar check?
                check_matches!(
                    (&self.rustc_abi, &self.cfg_abi),
                    (Some(RustcAbi::Softfloat), CfgAbi::SoftFloat)
                        | (
                            None,
                            CfgAbi::Ilp32
                                | CfgAbi::Llvm
                                | CfgAbi::MacAbi
                                | CfgAbi::Pauthtest
                                | CfgAbi::Sim
                                | CfgAbi::Uwp
                                | CfgAbi::Unspecified
                                | CfgAbi::Other(_)
                        ),
                    "invalid aarch64 Rust-specific ABI and `cfg(target_abi)` combination:\n\
                    Rust-specific ABI: {:?}\n\
                    cfg(target_abi): {}",
                    self.rustc_abi,
                    self.cfg_abi,
                );
            }
            Arch::PowerPC => {
                check!(
                    self.llvm_abiname == LlvmAbi::Unspecified,
                    "`llvm_abiname` is unused on PowerPC"
                );
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on PowerPC");
                check_matches!(
                    (&self.rustc_abi, &self.cfg_abi),
                    (Some(RustcAbi::PowerPcSpe), CfgAbi::Spe)
                        | (None, CfgAbi::Unspecified | CfgAbi::Other(_)),
                    "invalid PowerPC Rust-specific ABI and `cfg(target_abi)` combination:\n\
                    Rust-specific ABI: {:?}\n\
                    cfg(target_abi): {}",
                    self.rustc_abi,
                    self.cfg_abi,
                );
            }
            Arch::PowerPC64 => {
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on PowerPC64");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on PowerPC64");
                // PowerPC64 targets that are not AIX must set their ABI to either ELFv1 or ELFv2
                if self.os == Os::Aix {
                    // FIXME: Check that `target_abi` matches the actually configured ABI
                    // (vec-default vs vec-ext).
                    check_matches!(
                        (&self.llvm_abiname, &self.cfg_abi),
                        (LlvmAbi::Unspecified, CfgAbi::VecDefault | CfgAbi::VecExtAbi),
                        "invalid PowerPC64 AIX ABI name and `cfg(target_abi)` combination:\n\
                        ABI name: {}\n\
                        cfg(target_abi): {}",
                        self.llvm_abiname,
                        self.cfg_abi,
                    );
                } else if self.endian == Endian::Big {
                    check_matches!(
                        (&self.llvm_abiname, &self.cfg_abi),
                        (LlvmAbi::ElfV1, CfgAbi::ElfV1) | (LlvmAbi::ElfV2, CfgAbi::ElfV2),
                        "invalid PowerPC64 big-endian ABI name and `cfg(target_abi)` combination:\n\
                        ABI name: {}\n\
                        cfg(target_abi): {}",
                        self.llvm_abiname,
                        self.cfg_abi,
                    );
                } else {
                    check_matches!(
                        (&self.llvm_abiname, &self.cfg_abi),
                        (LlvmAbi::ElfV2, CfgAbi::ElfV2),
                        "invalid PowerPC64 little-endian ABI name and `cfg(target_abi)` combination:\n\
                        ABI name: {}\n\
                        cfg(target_abi): {}",
                        self.llvm_abiname,
                        self.cfg_abi,
                    );
                }
            }
            Arch::S390x => {
                check!(
                    self.llvm_abiname == LlvmAbi::Unspecified,
                    "`llvm_abiname` is unused on s390x"
                );
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on s390x");
                check_matches!(
                    (&self.rustc_abi, &self.cfg_abi),
                    (Some(RustcAbi::Softfloat), CfgAbi::SoftFloat)
                        | (None, CfgAbi::Unspecified | CfgAbi::Other(_)),
                    "invalid s390x Rust-specific ABI and `cfg(target_abi)` combination:\n\
                    Rust-specific ABI: {:?}\n\
                    cfg(target_abi): {}",
                    self.rustc_abi,
                    self.cfg_abi,
                );
            }
            Arch::LoongArch32 => {
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on LoongArch");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on LoongArch");
                check_matches!(
                    (&self.llvm_abiname, &self.cfg_abi),
                    (LlvmAbi::Ilp32s, CfgAbi::SoftFloat)
                        | (LlvmAbi::Ilp32f, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Ilp32d, CfgAbi::Unspecified | CfgAbi::Other(_)),
                    "invalid LoongArch ABI name and `cfg(target_abi)` combination:\n\
                     ABI name: {}\n\
                     cfg(target_abi): {}",
                    self.llvm_abiname,
                    self.cfg_abi,
                );
            }
            Arch::LoongArch64 => {
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on LoongArch");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on LoongArch");
                check_matches!(
                    (&self.llvm_abiname, &self.cfg_abi),
                    (LlvmAbi::Lp64s, CfgAbi::SoftFloat)
                        | (LlvmAbi::Lp64f, CfgAbi::Unspecified | CfgAbi::Other(_))
                        | (LlvmAbi::Lp64d, CfgAbi::Unspecified | CfgAbi::Other(_)),
                    "invalid LoongArch ABI name and `cfg(target_abi)` combination:\n\
                     ABI name: {}\n\
                     cfg(target_abi): {}",
                    self.llvm_abiname,
                    self.cfg_abi,
                );
            }
            Arch::Mips | Arch::Mips32r6 => {
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on MIPS");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on MIPS");
                check_matches!(
                    (&self.llvm_abiname, &self.cfg_abi),
                    (LlvmAbi::O32, CfgAbi::Unspecified | CfgAbi::Other(_)),
                    "invalid MIPS ABI name and `cfg(target_abi)` combination:\n\
                     ABI name: {}\n\
                     cfg(target_abi): {}",
                    self.llvm_abiname,
                    self.cfg_abi,
                );
            }
            Arch::Mips64 | Arch::Mips64r6 => {
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on MIPS");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on MIPS");
                check_matches!(
                    (&self.llvm_abiname, &self.cfg_abi),
                    // No in-tree targets use "n32" but at least for now we let out-of-tree targets
                    // experiment with that.
                    (LlvmAbi::N64, CfgAbi::Abi64)
                        | (LlvmAbi::N32, CfgAbi::Unspecified | CfgAbi::Other(_)),
                    "invalid MIPS ABI name and `cfg(target_abi)` combination:\n\
                     ABI name: {}\n\
                     cfg(target_abi): {}",
                    self.llvm_abiname,
                    self.cfg_abi,
                );
            }
            Arch::CSky => {
                check!(
                    self.llvm_abiname == LlvmAbi::Unspecified,
                    "`llvm_abiname` is unused on CSky"
                );
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on CSky");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on CSky");
                // FIXME: Check that `target_abi` matches the actually configured ABI (v2 vs v2hf).
                check_matches!(
                    self.cfg_abi,
                    CfgAbi::AbiV2 | CfgAbi::AbiV2Hf,
                    "invalid `target_abi` for CSky"
                );
            }
            Arch::Wasm32 | Arch::Wasm64 => {
                check!(
                    self.llvm_abiname == LlvmAbi::Unspecified,
                    "`llvm_abiname` is unused on wasm"
                );
                check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on wasm");
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on wasm");
                check_matches!(
                    self.cfg_abi,
                    CfgAbi::Unspecified | CfgAbi::Other(_),
                    "invalid `target_abi` for wasm"
                );
            }
            ref arch => {
                check!(self.rustc_abi.is_none(), "`rustc_abi` is unused on {arch}");
                // Ensure consistency among built-in targets, but give JSON targets the opportunity
                // to experiment with these.
                if kind == TargetKind::Builtin {
                    check!(
                        self.llvm_abiname == LlvmAbi::Unspecified,
                        "`llvm_abiname` is unused on {arch}"
                    );
                    check!(self.llvm_floatabi.is_none(), "`llvm_floatabi` is unused on {arch}");
                    check_matches!(
                        self.cfg_abi,
                        CfgAbi::Unspecified | CfgAbi::Other(_),
                        "`target_abi` is unused on {arch}"
                    );
                }
            }
        }

        // Check that the target cpu constraints make sense.
        if self.need_explicit_cpu {
            check!(
                self.requires_consistent_cpu,
                "if `need_explicit_cpu` is set, then `requires_consistent_cpu` must be set"
            );
        }

        // Check that the given target-features string makes some basic sense.
        if !self.features.is_empty() {
            let mut features_enabled = FxHashSet::default();
            let mut features_disabled = FxHashSet::default();
            for feat in self.features.split(',') {
                if let Some(feat) = feat.strip_prefix("+") {
                    features_enabled.insert(feat);
                    if features_disabled.contains(feat) {
                        return Err(format!(
                            "target feature `{feat}` is both enabled and disabled"
                        ));
                    }
                } else if let Some(feat) = feat.strip_prefix("-") {
                    features_disabled.insert(feat);
                    if features_enabled.contains(feat) {
                        return Err(format!(
                            "target feature `{feat}` is both enabled and disabled"
                        ));
                    }
                } else {
                    return Err(format!(
                        "target feature `{feat}` is invalid, must start with `+` or `-`"
                    ));
                }
            }
            // Check that we don't mis-set any of the ABI-relevant features.
            let abi_feature_constraints = self.abi_required_features();
            for feat in abi_feature_constraints.required {
                // The feature might be enabled by default so we can't *require* it to show up.
                // But it must not be *disabled*.
                if features_disabled.contains(feat) {
                    return Err(format!(
                        "target feature `{feat}` is required by the ABI but gets disabled in target spec"
                    ));
                }
            }
            for feat in abi_feature_constraints.incompatible {
                // The feature might be disabled by default so we can't *require* it to show up.
                // But it must not be *enabled*.
                if features_enabled.contains(feat) {
                    return Err(format!(
                        "target feature `{feat}` is incompatible with the ABI but gets enabled in target spec"
                    ));
                }
            }
        }

        Ok(())
    }
}
