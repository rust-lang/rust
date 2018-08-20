// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! [Flexible target specification.](https://github.com/rust-lang/rfcs/pull/131)
//!
//! Rust targets a wide variety of usecases, and in the interest of flexibility,
//! allows new target triples to be defined in configuration files. Most users
//! will not need to care about these, but this is invaluable when porting Rust
//! to a new platform, and allows for an unprecedented level of control over how
//! the compiler works.
//!
//! # Using custom targets
//!
//! A target triple, as passed via `rustc --target=TRIPLE`, will first be
//! compared against the list of built-in targets. This is to ease distributing
//! rustc (no need for configuration files) and also to hold these built-in
//! targets as immutable and sacred. If `TRIPLE` is not one of the built-in
//! targets, rustc will check if a file named `TRIPLE` exists. If it does, it
//! will be loaded as the target configuration. If the file does not exist,
//! rustc will search each directory in the environment variable
//! `RUST_TARGET_PATH` for a file named `TRIPLE.json`. The first one found will
//! be loaded. If no file is found in any of those directories, a fatal error
//! will be given.
//!
//! Projects defining their own targets should use
//! `--target=path/to/my-awesome-platform.json` instead of adding to
//! `RUST_TARGET_PATH`.
//!
//! # Defining a new target
//!
//! Targets are defined using [JSON](http://json.org/). The `Target` struct in
//! this module defines the format the JSON file should take, though each
//! underscore in the field names should be replaced with a hyphen (`-`) in the
//! JSON file. Some fields are required in every target specification, such as
//! `llvm-target`, `target-endian`, `target-pointer-width`, `data-layout`,
//! `arch`, and `os`. In general, options passed to rustc with `-C` override
//! the target's settings, though `target-feature` and `link-args` will *add*
//! to the list specified by the target, rather than replace.

use serialize::json::{Json, ToJson};
use std::collections::BTreeMap;
use std::default::Default;
use std::{fmt, io};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use spec::abi::{Abi, lookup as lookup_abi};

pub mod abi;
mod android_base;
mod apple_base;
mod apple_ios_base;
mod arm_base;
mod bitrig_base;
mod cloudabi_base;
mod dragonfly_base;
mod freebsd_base;
mod haiku_base;
mod hermit_base;
mod linux_base;
mod linux_musl_base;
mod openbsd_base;
mod netbsd_base;
mod solaris_base;
mod windows_base;
mod windows_msvc_base;
mod thumb_base;
mod l4re_base;
mod fuchsia_base;
mod redox_base;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash,
         RustcEncodable, RustcDecodable)]
pub enum LinkerFlavor {
    Em,
    Gcc,
    Ld,
    Msvc,
    Lld(LldFlavor),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash,
         RustcEncodable, RustcDecodable)]
pub enum LldFlavor {
    Wasm,
    Ld64,
    Ld,
    Link,
}

impl LldFlavor {
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "darwin" => LldFlavor::Ld64,
            "gnu" => LldFlavor::Ld,
            "link" => LldFlavor::Link,
            "wasm" => LldFlavor::Wasm,
            _ => return None,
        })
    }
}

impl ToJson for LldFlavor {
    fn to_json(&self) -> Json {
        match *self {
            LldFlavor::Ld64 => "darwin",
            LldFlavor::Ld => "gnu",
            LldFlavor::Link => "link",
            LldFlavor::Wasm => "wasm",
        }.to_json()
    }
}

impl ToJson for LinkerFlavor {
    fn to_json(&self) -> Json {
        self.desc().to_json()
    }
}
macro_rules! flavor_mappings {
    ($((($($flavor:tt)*), $string:expr),)*) => (
        impl LinkerFlavor {
            pub const fn one_of() -> &'static str {
                concat!("one of: ", $($string, " ",)+)
            }

            pub fn from_str(s: &str) -> Option<Self> {
                Some(match s {
                    $($string => $($flavor)*,)+
                    _ => return None,
                })
            }

            pub fn desc(&self) -> &str {
                match *self {
                    $($($flavor)* => $string,)+
                }
            }
        }
    )
}


flavor_mappings! {
    ((LinkerFlavor::Em), "em"),
    ((LinkerFlavor::Gcc), "gcc"),
    ((LinkerFlavor::Ld), "ld"),
    ((LinkerFlavor::Msvc), "msvc"),
    ((LinkerFlavor::Lld(LldFlavor::Wasm)), "wasm-ld"),
    ((LinkerFlavor::Lld(LldFlavor::Ld64)), "ld64.lld"),
    ((LinkerFlavor::Lld(LldFlavor::Ld)), "ld.lld"),
    ((LinkerFlavor::Lld(LldFlavor::Link)), "lld-link"),
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub enum PanicStrategy {
    Unwind,
    Abort,
}

impl PanicStrategy {
    pub fn desc(&self) -> &str {
        match *self {
            PanicStrategy::Unwind => "unwind",
            PanicStrategy::Abort => "abort",
        }
    }
}

impl ToJson for PanicStrategy {
    fn to_json(&self) -> Json {
        match *self {
            PanicStrategy::Abort => "abort".to_json(),
            PanicStrategy::Unwind => "unwind".to_json(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub enum RelroLevel {
    Full,
    Partial,
    Off,
    None,
}

impl RelroLevel {
    pub fn desc(&self) -> &str {
        match *self {
            RelroLevel::Full => "full",
            RelroLevel::Partial => "partial",
            RelroLevel::Off => "off",
            RelroLevel::None => "none",
        }
    }
}

impl FromStr for RelroLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<RelroLevel, ()> {
        match s {
            "full" => Ok(RelroLevel::Full),
            "partial" => Ok(RelroLevel::Partial),
            "off" => Ok(RelroLevel::Off),
            "none" => Ok(RelroLevel::None),
            _ => Err(()),
        }
    }
}

impl ToJson for RelroLevel {
    fn to_json(&self) -> Json {
        match *self {
            RelroLevel::Full => "full".to_json(),
            RelroLevel::Partial => "partial".to_json(),
            RelroLevel::Off => "off".to_json(),
            RelroLevel::None => "None".to_json(),
        }
    }
}

pub type LinkArgs = BTreeMap<LinkerFlavor, Vec<String>>;
pub type TargetResult = Result<Target, String>;

macro_rules! supported_targets {
    ( $(($triple:expr, $module:ident),)+ ) => (
        $(mod $module;)*

        /// List of supported targets
        const TARGETS: &'static [&'static str] = &[$($triple),*];

        fn load_specific(target: &str) -> TargetResult {
            match target {
                $(
                    $triple => {
                        let mut t = $module::target()?;
                        t.options.is_builtin = true;

                        // round-trip through the JSON parser to ensure at
                        // run-time that the parser works correctly
                        t = Target::from_json(t.to_json())?;
                        debug!("Got builtin target: {:?}", t);
                        Ok(t)
                    },
                )+
                _ => Err(format!("Unable to find target: {}", target))
            }
        }

        pub fn get_targets() -> Box<dyn Iterator<Item=String>> {
            Box::new(TARGETS.iter().filter_map(|t| -> Option<String> {
                load_specific(t)
                    .and(Ok(t.to_string()))
                    .ok()
            }))
        }

        #[cfg(test)]
        mod test_json_encode_decode {
            use serialize::json::ToJson;
            use super::Target;
            $(use super::$module;)*

            $(
                #[test]
                fn $module() {
                    // Grab the TargetResult struct. If we successfully retrieved
                    // a Target, then the test JSON encoding/decoding can run for this
                    // Target on this testing platform (i.e., checking the iOS targets
                    // only on a Mac test platform).
                    let _ = $module::target().map(|original| {
                        let as_json = original.to_json();
                        let parsed = Target::from_json(as_json).unwrap();
                        assert_eq!(original, parsed);
                    });
                }
            )*
        }
    )
}

supported_targets! {
    ("x86_64-unknown-linux-gnu", x86_64_unknown_linux_gnu),
    ("x86_64-unknown-linux-gnux32", x86_64_unknown_linux_gnux32),
    ("i686-unknown-linux-gnu", i686_unknown_linux_gnu),
    ("i586-unknown-linux-gnu", i586_unknown_linux_gnu),
    ("mips-unknown-linux-gnu", mips_unknown_linux_gnu),
    ("mips64-unknown-linux-gnuabi64", mips64_unknown_linux_gnuabi64),
    ("mips64el-unknown-linux-gnuabi64", mips64el_unknown_linux_gnuabi64),
    ("mipsel-unknown-linux-gnu", mipsel_unknown_linux_gnu),
    ("powerpc-unknown-linux-gnu", powerpc_unknown_linux_gnu),
    ("powerpc-unknown-linux-gnuspe", powerpc_unknown_linux_gnuspe),
    ("powerpc64-unknown-linux-gnu", powerpc64_unknown_linux_gnu),
    ("powerpc64le-unknown-linux-gnu", powerpc64le_unknown_linux_gnu),
    ("powerpc64le-unknown-linux-musl", powerpc64le_unknown_linux_musl),
    ("s390x-unknown-linux-gnu", s390x_unknown_linux_gnu),
    ("sparc-unknown-linux-gnu", sparc_unknown_linux_gnu),
    ("sparc64-unknown-linux-gnu", sparc64_unknown_linux_gnu),
    ("arm-unknown-linux-gnueabi", arm_unknown_linux_gnueabi),
    ("arm-unknown-linux-gnueabihf", arm_unknown_linux_gnueabihf),
    ("arm-unknown-linux-musleabi", arm_unknown_linux_musleabi),
    ("arm-unknown-linux-musleabihf", arm_unknown_linux_musleabihf),
    ("armv4t-unknown-linux-gnueabi", armv4t_unknown_linux_gnueabi),
    ("armv5te-unknown-linux-gnueabi", armv5te_unknown_linux_gnueabi),
    ("armv5te-unknown-linux-musleabi", armv5te_unknown_linux_musleabi),
    ("armv7-unknown-linux-gnueabihf", armv7_unknown_linux_gnueabihf),
    ("armv7-unknown-linux-musleabihf", armv7_unknown_linux_musleabihf),
    ("aarch64-unknown-linux-gnu", aarch64_unknown_linux_gnu),

    ("aarch64-unknown-linux-musl", aarch64_unknown_linux_musl),
    ("x86_64-unknown-linux-musl", x86_64_unknown_linux_musl),
    ("i686-unknown-linux-musl", i686_unknown_linux_musl),
    ("i586-unknown-linux-musl", i586_unknown_linux_musl),
    ("mips-unknown-linux-musl", mips_unknown_linux_musl),
    ("mipsel-unknown-linux-musl", mipsel_unknown_linux_musl),

    ("mips-unknown-linux-uclibc", mips_unknown_linux_uclibc),
    ("mipsel-unknown-linux-uclibc", mipsel_unknown_linux_uclibc),

    ("i686-linux-android", i686_linux_android),
    ("x86_64-linux-android", x86_64_linux_android),
    ("arm-linux-androideabi", arm_linux_androideabi),
    ("armv7-linux-androideabi", armv7_linux_androideabi),
    ("aarch64-linux-android", aarch64_linux_android),

    ("aarch64-unknown-freebsd", aarch64_unknown_freebsd),
    ("i686-unknown-freebsd", i686_unknown_freebsd),
    ("x86_64-unknown-freebsd", x86_64_unknown_freebsd),

    ("i686-unknown-dragonfly", i686_unknown_dragonfly),
    ("x86_64-unknown-dragonfly", x86_64_unknown_dragonfly),

    ("x86_64-unknown-bitrig", x86_64_unknown_bitrig),

    ("aarch64-unknown-openbsd", aarch64_unknown_openbsd),
    ("i686-unknown-openbsd", i686_unknown_openbsd),
    ("x86_64-unknown-openbsd", x86_64_unknown_openbsd),

    ("aarch64-unknown-netbsd", aarch64_unknown_netbsd),
    ("armv6-unknown-netbsd-eabihf", armv6_unknown_netbsd_eabihf),
    ("armv7-unknown-netbsd-eabihf", armv7_unknown_netbsd_eabihf),
    ("i686-unknown-netbsd", i686_unknown_netbsd),
    ("powerpc-unknown-netbsd", powerpc_unknown_netbsd),
    ("sparc64-unknown-netbsd", sparc64_unknown_netbsd),
    ("x86_64-unknown-netbsd", x86_64_unknown_netbsd),
    ("x86_64-rumprun-netbsd", x86_64_rumprun_netbsd),

    ("i686-unknown-haiku", i686_unknown_haiku),
    ("x86_64-unknown-haiku", x86_64_unknown_haiku),

    ("x86_64-apple-darwin", x86_64_apple_darwin),
    ("i686-apple-darwin", i686_apple_darwin),

    ("aarch64-fuchsia", aarch64_fuchsia),
    ("x86_64-fuchsia", x86_64_fuchsia),

    ("x86_64-unknown-l4re-uclibc", x86_64_unknown_l4re_uclibc),

    ("x86_64-unknown-redox", x86_64_unknown_redox),

    ("i386-apple-ios", i386_apple_ios),
    ("x86_64-apple-ios", x86_64_apple_ios),
    ("aarch64-apple-ios", aarch64_apple_ios),
    ("armv7-apple-ios", armv7_apple_ios),
    ("armv7s-apple-ios", armv7s_apple_ios),

    ("armebv7r-none-eabihf", armebv7r_none_eabihf),

    ("x86_64-sun-solaris", x86_64_sun_solaris),
    ("sparcv9-sun-solaris", sparcv9_sun_solaris),

    ("x86_64-pc-windows-gnu", x86_64_pc_windows_gnu),
    ("i686-pc-windows-gnu", i686_pc_windows_gnu),

    ("aarch64-pc-windows-msvc", aarch64_pc_windows_msvc),
    ("x86_64-pc-windows-msvc", x86_64_pc_windows_msvc),
    ("i686-pc-windows-msvc", i686_pc_windows_msvc),
    ("i586-pc-windows-msvc", i586_pc_windows_msvc),

    ("asmjs-unknown-emscripten", asmjs_unknown_emscripten),
    ("wasm32-unknown-emscripten", wasm32_unknown_emscripten),
    ("wasm32-unknown-unknown", wasm32_unknown_unknown),
    ("wasm32-experimental-emscripten", wasm32_experimental_emscripten),

    ("thumbv6m-none-eabi", thumbv6m_none_eabi),
    ("thumbv7m-none-eabi", thumbv7m_none_eabi),
    ("thumbv7em-none-eabi", thumbv7em_none_eabi),
    ("thumbv7em-none-eabihf", thumbv7em_none_eabihf),

    ("msp430-none-elf", msp430_none_elf),

    ("aarch64-unknown-cloudabi", aarch64_unknown_cloudabi),
    ("armv7-unknown-cloudabi-eabihf", armv7_unknown_cloudabi_eabihf),
    ("i686-unknown-cloudabi", i686_unknown_cloudabi),
    ("x86_64-unknown-cloudabi", x86_64_unknown_cloudabi),

    ("aarch64-unknown-hermit", aarch64_unknown_hermit),
    ("x86_64-unknown-hermit", x86_64_unknown_hermit),

    ("riscv32imac-unknown-none-elf", riscv32imac_unknown_none_elf),

    ("aarch64-unknown-none", aarch64_unknown_none),
}

/// Everything `rustc` knows about how to compile for a specific target.
///
/// Every field here must be specified, and has no default value.
#[derive(PartialEq, Clone, Debug)]
pub struct Target {
    /// Target triple to pass to LLVM.
    pub llvm_target: String,
    /// String to use as the `target_endian` `cfg` variable.
    pub target_endian: String,
    /// String to use as the `target_pointer_width` `cfg` variable.
    pub target_pointer_width: String,
    /// Width of c_int type
    pub target_c_int_width: String,
    /// OS name to use for conditional compilation.
    pub target_os: String,
    /// Environment name to use for conditional compilation.
    pub target_env: String,
    /// Vendor name to use for conditional compilation.
    pub target_vendor: String,
    /// Architecture to use for ABI considerations. Valid options: "x86",
    /// "x86_64", "arm", "aarch64", "mips", "powerpc", and "powerpc64".
    pub arch: String,
    /// [Data layout](http://llvm.org/docs/LangRef.html#data-layout) to pass to LLVM.
    pub data_layout: String,
    /// Linker flavor
    pub linker_flavor: LinkerFlavor,
    /// Optional settings with defaults.
    pub options: TargetOptions,
}

pub trait HasTargetSpec: Copy {
    fn target_spec(&self) -> &Target;
}

impl<'a> HasTargetSpec for &'a Target {
    fn target_spec(&self) -> &Target {
        self
    }
}

/// Optional aspects of a target specification.
///
/// This has an implementation of `Default`, see each field for what the default is. In general,
/// these try to take "minimal defaults" that don't assume anything about the runtime they run in.
#[derive(PartialEq, Clone, Debug)]
pub struct TargetOptions {
    /// Whether the target is built-in or loaded from a custom target specification.
    pub is_builtin: bool,

    /// Linker to invoke
    pub linker: Option<String>,

    /// LLD flavor
    pub lld_flavor: LldFlavor,

    /// Linker arguments that are passed *before* any user-defined libraries.
    pub pre_link_args: LinkArgs, // ... unconditionally
    pub pre_link_args_crt: LinkArgs, // ... when linking with a bundled crt
    /// Objects to link before all others, always found within the
    /// sysroot folder.
    pub pre_link_objects_exe: Vec<String>, // ... when linking an executable, unconditionally
    pub pre_link_objects_exe_crt: Vec<String>, // ... when linking an executable with a bundled crt
    pub pre_link_objects_dll: Vec<String>, // ... when linking a dylib
    /// Linker arguments that are unconditionally passed after any
    /// user-defined but before post_link_objects.  Standard platform
    /// libraries that should be always be linked to, usually go here.
    pub late_link_args: LinkArgs,
    /// Objects to link after all others, always found within the
    /// sysroot folder.
    pub post_link_objects: Vec<String>, // ... unconditionally
    pub post_link_objects_crt: Vec<String>, // ... when linking with a bundled crt
    /// Linker arguments that are unconditionally passed *after* any
    /// user-defined libraries.
    pub post_link_args: LinkArgs,

    /// Environment variables to be set before invoking the linker.
    pub link_env: Vec<(String, String)>,

    /// Extra arguments to pass to the external assembler (when used)
    pub asm_args: Vec<String>,

    /// Default CPU to pass to LLVM. Corresponds to `llc -mcpu=$cpu`. Defaults
    /// to "generic".
    pub cpu: String,
    /// Default target features to pass to LLVM. These features will *always* be
    /// passed, and cannot be disabled even via `-C`. Corresponds to `llc
    /// -mattr=$features`.
    pub features: String,
    /// Whether dynamic linking is available on this target. Defaults to false.
    pub dynamic_linking: bool,
    /// If dynamic linking is available, whether only cdylibs are supported.
    pub only_cdylib: bool,
    /// Whether executables are available on this target. iOS, for example, only allows static
    /// libraries. Defaults to false.
    pub executables: bool,
    /// Relocation model to use in object file. Corresponds to `llc
    /// -relocation-model=$relocation_model`. Defaults to "pic".
    pub relocation_model: String,
    /// Code model to use. Corresponds to `llc -code-model=$code_model`.
    pub code_model: Option<String>,
    /// TLS model to use. Options are "global-dynamic" (default), "local-dynamic", "initial-exec"
    /// and "local-exec". This is similar to the -ftls-model option in GCC/Clang.
    pub tls_model: String,
    /// Do not emit code that uses the "red zone", if the ABI has one. Defaults to false.
    pub disable_redzone: bool,
    /// Eliminate frame pointers from stack frames if possible. Defaults to true.
    pub eliminate_frame_pointer: bool,
    /// Emit each function in its own section. Defaults to true.
    pub function_sections: bool,
    /// String to prepend to the name of every dynamic library. Defaults to "lib".
    pub dll_prefix: String,
    /// String to append to the name of every dynamic library. Defaults to ".so".
    pub dll_suffix: String,
    /// String to append to the name of every executable.
    pub exe_suffix: String,
    /// String to prepend to the name of every static library. Defaults to "lib".
    pub staticlib_prefix: String,
    /// String to append to the name of every static library. Defaults to ".a".
    pub staticlib_suffix: String,
    /// OS family to use for conditional compilation. Valid options: "unix", "windows".
    pub target_family: Option<String>,
    /// Whether the target toolchain's ABI supports returning small structs as an integer.
    pub abi_return_struct_as_int: bool,
    /// Whether the target toolchain is like macOS's. Only useful for compiling against iOS/macOS,
    /// in particular running dsymutil and some other stuff like `-dead_strip`. Defaults to false.
    pub is_like_osx: bool,
    /// Whether the target toolchain is like Solaris's.
    /// Only useful for compiling against Illumos/Solaris,
    /// as they have a different set of linker flags. Defaults to false.
    pub is_like_solaris: bool,
    /// Whether the target toolchain is like Windows'. Only useful for compiling against Windows,
    /// only really used for figuring out how to find libraries, since Windows uses its own
    /// library naming convention. Defaults to false.
    pub is_like_windows: bool,
    pub is_like_msvc: bool,
    /// Whether the target toolchain is like Android's. Only useful for compiling against Android.
    /// Defaults to false.
    pub is_like_android: bool,
    /// Whether the target toolchain is like Emscripten's. Only useful for compiling with
    /// Emscripten toolchain.
    /// Defaults to false.
    pub is_like_emscripten: bool,
    /// Whether the linker support GNU-like arguments such as -O. Defaults to false.
    pub linker_is_gnu: bool,
    /// The MinGW toolchain has a known issue that prevents it from correctly
    /// handling COFF object files with more than 2<sup>15</sup> sections. Since each weak
    /// symbol needs its own COMDAT section, weak linkage implies a large
    /// number sections that easily exceeds the given limit for larger
    /// codebases. Consequently we want a way to disallow weak linkage on some
    /// platforms.
    pub allows_weak_linkage: bool,
    /// Whether the linker support rpaths or not. Defaults to false.
    pub has_rpath: bool,
    /// Whether to disable linking to the default libraries, typically corresponds
    /// to `-nodefaultlibs`. Defaults to true.
    pub no_default_libraries: bool,
    /// Dynamically linked executables can be compiled as position independent
    /// if the default relocation model of position independent code is not
    /// changed. This is a requirement to take advantage of ASLR, as otherwise
    /// the functions in the executable are not randomized and can be used
    /// during an exploit of a vulnerability in any code.
    pub position_independent_executables: bool,
    /// Either partial, full, or off. Full RELRO makes the dynamic linker
    /// resolve all symbols at startup and marks the GOT read-only before
    /// starting the program, preventing overwriting the GOT.
    pub relro_level: RelroLevel,
    /// Format that archives should be emitted in. This affects whether we use
    /// LLVM to assemble an archive or fall back to the system linker, and
    /// currently only "gnu" is used to fall into LLVM. Unknown strings cause
    /// the system linker to be used.
    pub archive_format: String,
    /// Is asm!() allowed? Defaults to true.
    pub allow_asm: bool,
    /// Whether the target uses a custom unwind resumption routine.
    /// By default LLVM lowers `resume` instructions into calls to `_Unwind_Resume`
    /// defined in libgcc.  If this option is enabled, the target must provide
    /// `eh_unwind_resume` lang item.
    pub custom_unwind_resume: bool,

    /// If necessary, a different crate to link exe allocators by default
    pub exe_allocation_crate: Option<String>,

    /// Flag indicating whether ELF TLS (e.g. #[thread_local]) is available for
    /// this target.
    pub has_elf_tls: bool,
    // This is mainly for easy compatibility with emscripten.
    // If we give emcc .o files that are actually .bc files it
    // will 'just work'.
    pub obj_is_bitcode: bool,

    // LLVM can't produce object files for this target. Instead, we'll make LLVM
    // emit assembly and then use `gcc` to turn that assembly into an object
    // file
    pub no_integrated_as: bool,

    /// Don't use this field; instead use the `.min_atomic_width()` method.
    pub min_atomic_width: Option<u64>,

    /// Don't use this field; instead use the `.max_atomic_width()` method.
    pub max_atomic_width: Option<u64>,

    /// Whether the target supports atomic CAS operations natively
    pub atomic_cas: bool,

    /// Panic strategy: "unwind" or "abort"
    pub panic_strategy: PanicStrategy,

    /// A blacklist of ABIs unsupported by the current target. Note that generic
    /// ABIs are considered to be supported on all platforms and cannot be blacklisted.
    pub abi_blacklist: Vec<Abi>,

    /// Whether or not linking dylibs to a static CRT is allowed.
    pub crt_static_allows_dylibs: bool,
    /// Whether or not the CRT is statically linked by default.
    pub crt_static_default: bool,
    /// Whether or not crt-static is respected by the compiler (or is a no-op).
    pub crt_static_respected: bool,

    /// Whether or not stack probes (__rust_probestack) are enabled
    pub stack_probes: bool,

    /// The minimum alignment for global symbols.
    pub min_global_align: Option<u64>,

    /// Default number of codegen units to use in debug mode
    pub default_codegen_units: Option<u64>,

    /// Whether to generate trap instructions in places where optimization would
    /// otherwise produce control flow that falls through into unrelated memory.
    pub trap_unreachable: bool,

    /// This target requires everything to be compiled with LTO to emit a final
    /// executable, aka there is no native linker for this target.
    pub requires_lto: bool,

    /// This target has no support for threads.
    pub singlethread: bool,

    /// Whether library functions call lowering/optimization is disabled in LLVM
    /// for this target unconditionally.
    pub no_builtins: bool,

    /// Whether to lower 128-bit operations to compiler_builtins calls.  Use if
    /// your backend only supports 64-bit and smaller math.
    pub i128_lowering: bool,

    /// The codegen backend to use for this target, typically "llvm"
    pub codegen_backend: String,

    /// The default visibility for symbols in this target should be "hidden"
    /// rather than "default"
    pub default_hidden_visibility: bool,

    /// Whether or not bitcode is embedded in object files
    pub embed_bitcode: bool,

    /// Whether a .debug_gdb_scripts section will be added to the output object file
    pub emit_debug_gdb_scripts: bool,

    /// Whether or not to unconditionally `uwtable` attributes on functions,
    /// typically because the platform needs to unwind for things like stack
    /// unwinders.
    pub requires_uwtable: bool,
}

impl Default for TargetOptions {
    /// Create a set of "sane defaults" for any target. This is still
    /// incomplete, and if used for compilation, will certainly not work.
    fn default() -> TargetOptions {
        TargetOptions {
            is_builtin: false,
            linker: option_env!("CFG_DEFAULT_LINKER").map(|s| s.to_string()),
            lld_flavor: LldFlavor::Ld,
            pre_link_args: LinkArgs::new(),
            pre_link_args_crt: LinkArgs::new(),
            post_link_args: LinkArgs::new(),
            asm_args: Vec::new(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            dynamic_linking: false,
            only_cdylib: false,
            executables: false,
            relocation_model: "pic".to_string(),
            code_model: None,
            tls_model: "global-dynamic".to_string(),
            disable_redzone: false,
            eliminate_frame_pointer: true,
            function_sections: true,
            dll_prefix: "lib".to_string(),
            dll_suffix: ".so".to_string(),
            exe_suffix: "".to_string(),
            staticlib_prefix: "lib".to_string(),
            staticlib_suffix: ".a".to_string(),
            target_family: None,
            abi_return_struct_as_int: false,
            is_like_osx: false,
            is_like_solaris: false,
            is_like_windows: false,
            is_like_android: false,
            is_like_emscripten: false,
            is_like_msvc: false,
            linker_is_gnu: false,
            allows_weak_linkage: true,
            has_rpath: false,
            no_default_libraries: true,
            position_independent_executables: false,
            relro_level: RelroLevel::None,
            pre_link_objects_exe: Vec::new(),
            pre_link_objects_exe_crt: Vec::new(),
            pre_link_objects_dll: Vec::new(),
            post_link_objects: Vec::new(),
            post_link_objects_crt: Vec::new(),
            late_link_args: LinkArgs::new(),
            link_env: Vec::new(),
            archive_format: "gnu".to_string(),
            custom_unwind_resume: false,
            exe_allocation_crate: None,
            allow_asm: true,
            has_elf_tls: false,
            obj_is_bitcode: false,
            no_integrated_as: false,
            min_atomic_width: None,
            max_atomic_width: None,
            atomic_cas: true,
            panic_strategy: PanicStrategy::Unwind,
            abi_blacklist: vec![],
            crt_static_allows_dylibs: false,
            crt_static_default: false,
            crt_static_respected: false,
            stack_probes: false,
            min_global_align: None,
            default_codegen_units: None,
            trap_unreachable: true,
            requires_lto: false,
            singlethread: false,
            no_builtins: false,
            i128_lowering: false,
            codegen_backend: "llvm".to_string(),
            default_hidden_visibility: false,
            embed_bitcode: false,
            emit_debug_gdb_scripts: true,
            requires_uwtable: false,
        }
    }
}

impl Target {
    /// Given a function ABI, turn "System" into the correct ABI for this target.
    pub fn adjust_abi(&self, abi: Abi) -> Abi {
        match abi {
            Abi::System => {
                if self.options.is_like_windows && self.arch == "x86" {
                    Abi::Stdcall
                } else {
                    Abi::C
                }
            },
            abi => abi
        }
    }

    /// Minimum integer size in bits that this target can perform atomic
    /// operations on.
    pub fn min_atomic_width(&self) -> u64 {
        self.options.min_atomic_width.unwrap_or(8)
    }

    /// Maximum integer size in bits that this target can perform atomic
    /// operations on.
    pub fn max_atomic_width(&self) -> u64 {
        self.options.max_atomic_width.unwrap_or_else(|| self.target_pointer_width.parse().unwrap())
    }

    pub fn is_abi_supported(&self, abi: Abi) -> bool {
        abi.generic() || !self.options.abi_blacklist.contains(&abi)
    }

    /// Load a target descriptor from a JSON object.
    pub fn from_json(obj: Json) -> TargetResult {
        // While ugly, this code must remain this way to retain
        // compatibility with existing JSON fields and the internal
        // expected naming of the Target and TargetOptions structs.
        // To ensure compatibility is retained, the built-in targets
        // are round-tripped through this code to catch cases where
        // the JSON parser is not updated to match the structs.

        let get_req_field = |name: &str| {
            obj.find(name)
               .map(|s| s.as_string())
               .and_then(|os| os.map(|s| s.to_string()))
               .ok_or_else(|| format!("Field {} in target specification is required", name))
        };

        let get_opt_field = |name: &str, default: &str| {
            obj.find(name).and_then(|s| s.as_string())
               .map(|s| s.to_string())
               .unwrap_or_else(|| default.to_string())
        };

        let mut base = Target {
            llvm_target: get_req_field("llvm-target")?,
            target_endian: get_req_field("target-endian")?,
            target_pointer_width: get_req_field("target-pointer-width")?,
            target_c_int_width: get_req_field("target-c-int-width")?,
            data_layout: get_req_field("data-layout")?,
            arch: get_req_field("arch")?,
            target_os: get_req_field("os")?,
            target_env: get_opt_field("env", ""),
            target_vendor: get_opt_field("vendor", "unknown"),
            linker_flavor: LinkerFlavor::from_str(&*get_req_field("linker-flavor")?)
                .ok_or_else(|| {
                    format!("linker flavor must be {}", LinkerFlavor::one_of())
                })?,
            options: Default::default(),
        };

        macro_rules! key {
            ($key_name:ident) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..]).map(|o| o.as_string()
                                    .map(|s| base.options.$key_name = s.to_string()));
            } );
            ($key_name:ident, bool) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..])
                    .map(|o| o.as_boolean()
                         .map(|s| base.options.$key_name = s));
            } );
            ($key_name:ident, Option<u64>) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..])
                    .map(|o| o.as_u64()
                         .map(|s| base.options.$key_name = Some(s)));
            } );
            ($key_name:ident, PanicStrategy) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..]).and_then(|o| o.as_string().and_then(|s| {
                    match s {
                        "unwind" => base.options.$key_name = PanicStrategy::Unwind,
                        "abort" => base.options.$key_name = PanicStrategy::Abort,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      panic-strategy. Use 'unwind' or 'abort'.",
                                                     s))),
                }
                Some(Ok(()))
            })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, RelroLevel) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..]).and_then(|o| o.as_string().and_then(|s| {
                    match s.parse::<RelroLevel>() {
                        Ok(level) => base.options.$key_name = level,
                        _ => return Some(Err(format!("'{}' is not a valid value for \
                                                      relro-level. Use 'full', 'partial, or 'off'.",
                                                      s))),
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..]).map(|o| o.as_array()
                    .map(|v| base.options.$key_name = v.iter()
                        .map(|a| a.as_string().unwrap().to_string()).collect()
                        )
                    );
            } );
            ($key_name:ident, optional) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(o) = obj.find(&name[..]) {
                    base.options.$key_name = o
                        .as_string()
                        .map(|s| s.to_string() );
                }
            } );
            ($key_name:ident, LldFlavor) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..]).and_then(|o| o.as_string().and_then(|s| {
                    if let Some(flavor) = LldFlavor::from_str(&s) {
                        base.options.$key_name = flavor;
                    } else {
                        return Some(Err(format!(
                            "'{}' is not a valid value for lld-flavor. \
                             Use 'darwin', 'gnu', 'link' or 'wasm.",
                            s)))
                    }
                    Some(Ok(()))
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, LinkerFlavor) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..]).and_then(|o| o.as_string().map(|s| {
                    LinkerFlavor::from_str(&s).ok_or_else(|| {
                        Err(format!("'{}' is not a valid value for linker-flavor. \
                                     Use 'em', 'gcc', 'ld' or 'msvc.", s))
                    })
                })).unwrap_or(Ok(()))
            } );
            ($key_name:ident, link_args) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(val) = obj.find(&name[..]) {
                    let obj = val.as_object().ok_or_else(|| format!("{}: expected a \
                        JSON object with fields per linker-flavor.", name))?;
                    let mut args = LinkArgs::new();
                    for (k, v) in obj {
                        let flavor = LinkerFlavor::from_str(&k).ok_or_else(|| {
                            format!("{}: '{}' is not a valid value for linker-flavor. \
                                     Use 'em', 'gcc', 'ld' or 'msvc'", name, k)
                        })?;

                        let v = v.as_array().ok_or_else(||
                            format!("{}.{}: expected a JSON array", name, k)
                        )?.iter().enumerate()
                            .map(|(i,s)| {
                                let s = s.as_string().ok_or_else(||
                                    format!("{}.{}[{}]: expected a JSON string", name, k, i))?;
                                Ok(s.to_owned())
                            })
                            .collect::<Result<Vec<_>, String>>()?;

                        args.insert(flavor, v);
                    }
                    base.options.$key_name = args;
                }
            } );
            ($key_name:ident, env) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                if let Some(a) = obj.find(&name[..]).and_then(|o| o.as_array()) {
                    for o in a {
                        if let Some(s) = o.as_string() {
                            let p = s.split('=').collect::<Vec<_>>();
                            if p.len() == 2 {
                                let k = p[0].to_string();
                                let v = p[1].to_string();
                                base.options.$key_name.push((k, v));
                            }
                        }
                    }
                }
            } );
        }

        key!(is_builtin, bool);
        key!(linker, optional);
        try!(key!(lld_flavor, LldFlavor));
        key!(pre_link_args, link_args);
        key!(pre_link_args_crt, link_args);
        key!(pre_link_objects_exe, list);
        key!(pre_link_objects_exe_crt, list);
        key!(pre_link_objects_dll, list);
        key!(late_link_args, link_args);
        key!(post_link_objects, list);
        key!(post_link_objects_crt, list);
        key!(post_link_args, link_args);
        key!(link_env, env);
        key!(asm_args, list);
        key!(cpu);
        key!(features);
        key!(dynamic_linking, bool);
        key!(only_cdylib, bool);
        key!(executables, bool);
        key!(relocation_model);
        key!(code_model, optional);
        key!(tls_model);
        key!(disable_redzone, bool);
        key!(eliminate_frame_pointer, bool);
        key!(function_sections, bool);
        key!(dll_prefix);
        key!(dll_suffix);
        key!(exe_suffix);
        key!(staticlib_prefix);
        key!(staticlib_suffix);
        key!(target_family, optional);
        key!(abi_return_struct_as_int, bool);
        key!(is_like_osx, bool);
        key!(is_like_solaris, bool);
        key!(is_like_windows, bool);
        key!(is_like_msvc, bool);
        key!(is_like_emscripten, bool);
        key!(is_like_android, bool);
        key!(linker_is_gnu, bool);
        key!(allows_weak_linkage, bool);
        key!(has_rpath, bool);
        key!(no_default_libraries, bool);
        key!(position_independent_executables, bool);
        try!(key!(relro_level, RelroLevel));
        key!(archive_format);
        key!(allow_asm, bool);
        key!(custom_unwind_resume, bool);
        key!(exe_allocation_crate, optional);
        key!(has_elf_tls, bool);
        key!(obj_is_bitcode, bool);
        key!(no_integrated_as, bool);
        key!(max_atomic_width, Option<u64>);
        key!(min_atomic_width, Option<u64>);
        key!(atomic_cas, bool);
        try!(key!(panic_strategy, PanicStrategy));
        key!(crt_static_allows_dylibs, bool);
        key!(crt_static_default, bool);
        key!(crt_static_respected, bool);
        key!(stack_probes, bool);
        key!(min_global_align, Option<u64>);
        key!(default_codegen_units, Option<u64>);
        key!(trap_unreachable, bool);
        key!(requires_lto, bool);
        key!(singlethread, bool);
        key!(no_builtins, bool);
        key!(codegen_backend);
        key!(default_hidden_visibility, bool);
        key!(embed_bitcode, bool);
        key!(emit_debug_gdb_scripts, bool);
        key!(requires_uwtable, bool);

        if let Some(array) = obj.find("abi-blacklist").and_then(Json::as_array) {
            for name in array.iter().filter_map(|abi| abi.as_string()) {
                match lookup_abi(name) {
                    Some(abi) => {
                        if abi.generic() {
                            return Err(format!("The ABI \"{}\" is considered to be supported on \
                                                all targets and cannot be blacklisted", abi))
                        }

                        base.options.abi_blacklist.push(abi)
                    }
                    None => return Err(format!("Unknown ABI \"{}\" in target specification", name))
                }
            }
        }

        Ok(base)
    }

    /// Search RUST_TARGET_PATH for a JSON file specifying the given target
    /// triple. Note that it could also just be a bare filename already, so also
    /// check for that. If one of the hardcoded targets we know about, just
    /// return it directly.
    ///
    /// The error string could come from any of the APIs called, including
    /// filesystem access and JSON decoding.
    pub fn search(target_triple: &TargetTriple) -> Result<Target, String> {
        use std::env;
        use std::fs;
        use serialize::json;

        fn load_file(path: &Path) -> Result<Target, String> {
            let contents = fs::read(path).map_err(|e| e.to_string())?;
            let obj = json::from_reader(&mut &contents[..])
                           .map_err(|e| e.to_string())?;
            Target::from_json(obj)
        }

        match *target_triple {
            TargetTriple::TargetTriple(ref target_triple) => {
                // check if triple is in list of supported targets
                if let Ok(t) = load_specific(target_triple) {
                    return Ok(t)
                }

                // search for a file named `target_triple`.json in RUST_TARGET_PATH
                let path = {
                    let mut target = target_triple.to_string();
                    target.push_str(".json");
                    PathBuf::from(target)
                };

                let target_path = env::var_os("RUST_TARGET_PATH").unwrap_or_default();

                // FIXME 16351: add a sane default search path?

                for dir in env::split_paths(&target_path) {
                    let p =  dir.join(&path);
                    if p.is_file() {
                        return load_file(&p);
                    }
                }
                Err(format!("Could not find specification for target {:?}", target_triple))
            }
            TargetTriple::TargetPath(ref target_path) => {
                if target_path.is_file() {
                    return load_file(&target_path);
                }
                Err(format!("Target path {:?} is not a valid file", target_path))
            }
        }
    }
}

impl ToJson for Target {
    fn to_json(&self) -> Json {
        let mut d = BTreeMap::new();
        let default: TargetOptions = Default::default();

        macro_rules! target_val {
            ($attr:ident) => ( {
                let name = (stringify!($attr)).replace("_", "-");
                d.insert(name.to_string(), self.$attr.to_json());
            } );
            ($attr:ident, $key_name:expr) => ( {
                let name = $key_name;
                d.insert(name.to_string(), self.$attr.to_json());
            } );
        }

        macro_rules! target_option_val {
            ($attr:ident) => ( {
                let name = (stringify!($attr)).replace("_", "-");
                if default.$attr != self.options.$attr {
                    d.insert(name.to_string(), self.options.$attr.to_json());
                }
            } );
            ($attr:ident, $key_name:expr) => ( {
                let name = $key_name;
                if default.$attr != self.options.$attr {
                    d.insert(name.to_string(), self.options.$attr.to_json());
                }
            } );
            (link_args - $attr:ident) => ( {
                let name = (stringify!($attr)).replace("_", "-");
                if default.$attr != self.options.$attr {
                    let obj = self.options.$attr
                        .iter()
                        .map(|(k, v)| (k.desc().to_owned(), v.clone()))
                        .collect::<BTreeMap<_, _>>();
                    d.insert(name.to_string(), obj.to_json());
                }
            } );
            (env - $attr:ident) => ( {
                let name = (stringify!($attr)).replace("_", "-");
                if default.$attr != self.options.$attr {
                    let obj = self.options.$attr
                        .iter()
                        .map(|&(ref k, ref v)| k.clone() + "=" + &v)
                        .collect::<Vec<_>>();
                    d.insert(name.to_string(), obj.to_json());
                }
            } );

        }

        target_val!(llvm_target);
        target_val!(target_endian);
        target_val!(target_pointer_width);
        target_val!(target_c_int_width);
        target_val!(arch);
        target_val!(target_os, "os");
        target_val!(target_env, "env");
        target_val!(target_vendor, "vendor");
        target_val!(data_layout);
        target_val!(linker_flavor);

        target_option_val!(is_builtin);
        target_option_val!(linker);
        target_option_val!(lld_flavor);
        target_option_val!(link_args - pre_link_args);
        target_option_val!(link_args - pre_link_args_crt);
        target_option_val!(pre_link_objects_exe);
        target_option_val!(pre_link_objects_exe_crt);
        target_option_val!(pre_link_objects_dll);
        target_option_val!(link_args - late_link_args);
        target_option_val!(post_link_objects);
        target_option_val!(post_link_objects_crt);
        target_option_val!(link_args - post_link_args);
        target_option_val!(env - link_env);
        target_option_val!(asm_args);
        target_option_val!(cpu);
        target_option_val!(features);
        target_option_val!(dynamic_linking);
        target_option_val!(only_cdylib);
        target_option_val!(executables);
        target_option_val!(relocation_model);
        target_option_val!(code_model);
        target_option_val!(tls_model);
        target_option_val!(disable_redzone);
        target_option_val!(eliminate_frame_pointer);
        target_option_val!(function_sections);
        target_option_val!(dll_prefix);
        target_option_val!(dll_suffix);
        target_option_val!(exe_suffix);
        target_option_val!(staticlib_prefix);
        target_option_val!(staticlib_suffix);
        target_option_val!(target_family);
        target_option_val!(abi_return_struct_as_int);
        target_option_val!(is_like_osx);
        target_option_val!(is_like_solaris);
        target_option_val!(is_like_windows);
        target_option_val!(is_like_msvc);
        target_option_val!(is_like_emscripten);
        target_option_val!(is_like_android);
        target_option_val!(linker_is_gnu);
        target_option_val!(allows_weak_linkage);
        target_option_val!(has_rpath);
        target_option_val!(no_default_libraries);
        target_option_val!(position_independent_executables);
        target_option_val!(relro_level);
        target_option_val!(archive_format);
        target_option_val!(allow_asm);
        target_option_val!(custom_unwind_resume);
        target_option_val!(exe_allocation_crate);
        target_option_val!(has_elf_tls);
        target_option_val!(obj_is_bitcode);
        target_option_val!(no_integrated_as);
        target_option_val!(min_atomic_width);
        target_option_val!(max_atomic_width);
        target_option_val!(atomic_cas);
        target_option_val!(panic_strategy);
        target_option_val!(crt_static_allows_dylibs);
        target_option_val!(crt_static_default);
        target_option_val!(crt_static_respected);
        target_option_val!(stack_probes);
        target_option_val!(min_global_align);
        target_option_val!(default_codegen_units);
        target_option_val!(trap_unreachable);
        target_option_val!(requires_lto);
        target_option_val!(singlethread);
        target_option_val!(no_builtins);
        target_option_val!(codegen_backend);
        target_option_val!(default_hidden_visibility);
        target_option_val!(embed_bitcode);
        target_option_val!(emit_debug_gdb_scripts);
        target_option_val!(requires_uwtable);

        if default.abi_blacklist != self.options.abi_blacklist {
            d.insert("abi-blacklist".to_string(), self.options.abi_blacklist.iter()
                .map(|&name| Abi::name(name).to_json())
                .collect::<Vec<_>>().to_json());
        }

        Json::Object(d)
    }
}

fn maybe_jemalloc() -> Option<String> {
    if cfg!(feature = "jemalloc") {
        Some("alloc_jemalloc".to_string())
    } else {
        None
    }
}

/// Either a target triple string or a path to a JSON file.
#[derive(PartialEq, Clone, Debug, Hash, RustcEncodable, RustcDecodable)]
pub enum TargetTriple {
    TargetTriple(String),
    TargetPath(PathBuf),
}

impl TargetTriple {
    /// Creates a target triple from the passed target triple string.
    pub fn from_triple(triple: &str) -> Self {
        TargetTriple::TargetTriple(triple.to_string())
    }

    /// Creates a target triple from the passed target path.
    pub fn from_path(path: &Path) -> Result<Self, io::Error> {
        let canonicalized_path = path.canonicalize()?;
        Ok(TargetTriple::TargetPath(canonicalized_path))
    }

    /// Returns a string triple for this target.
    ///
    /// If this target is a path, the file name (without extension) is returned.
    pub fn triple(&self) -> &str {
        match *self {
            TargetTriple::TargetTriple(ref triple) => triple,
            TargetTriple::TargetPath(ref path) => {
                path.file_stem().expect("target path must not be empty").to_str()
                    .expect("target path must be valid unicode")
            }
        }
    }

    /// Returns an extended string triple for this target.
    ///
    /// If this target is a path, a hash of the path is appended to the triple returned
    /// by `triple()`.
    pub fn debug_triple(&self) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let triple = self.triple();
        if let TargetTriple::TargetPath(ref path) = *self {
            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);
            let hash = hasher.finish();
            format!("{}-{}", triple, hash)
        } else {
            triple.to_owned()
        }
    }
}

impl fmt::Display for TargetTriple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.debug_triple())
    }
}
