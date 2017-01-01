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
use std::io::prelude::*;
use syntax::abi::{Abi, lookup as lookup_abi};

use PanicStrategy;

mod android_base;
mod apple_base;
mod apple_ios_base;
mod arm_base;
mod bitrig_base;
mod dragonfly_base;
mod emscripten_base;
mod freebsd_base;
mod haiku_base;
mod linux_base;
mod linux_musl_base;
mod openbsd_base;
mod netbsd_base;
mod solaris_base;
mod windows_base;
mod windows_msvc_base;
mod thumb_base;
mod fuchsia_base;
mod redox_base;

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

        pub fn get_targets() -> Box<Iterator<Item=String>> {
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
    ("i686-unknown-linux-gnu", i686_unknown_linux_gnu),
    ("i586-unknown-linux-gnu", i586_unknown_linux_gnu),
    ("mips-unknown-linux-gnu", mips_unknown_linux_gnu),
    ("mips64-unknown-linux-gnuabi64", mips64_unknown_linux_gnuabi64),
    ("mips64el-unknown-linux-gnuabi64", mips64el_unknown_linux_gnuabi64),
    ("mipsel-unknown-linux-gnu", mipsel_unknown_linux_gnu),
    ("powerpc-unknown-linux-gnu", powerpc_unknown_linux_gnu),
    ("powerpc64-unknown-linux-gnu", powerpc64_unknown_linux_gnu),
    ("powerpc64le-unknown-linux-gnu", powerpc64le_unknown_linux_gnu),
    ("s390x-unknown-linux-gnu", s390x_unknown_linux_gnu),
    ("arm-unknown-linux-gnueabi", arm_unknown_linux_gnueabi),
    ("arm-unknown-linux-gnueabihf", arm_unknown_linux_gnueabihf),
    ("arm-unknown-linux-musleabi", arm_unknown_linux_musleabi),
    ("arm-unknown-linux-musleabihf", arm_unknown_linux_musleabihf),
    ("armv5te-unknown-linux-gnueabi", armv5te_unknown_linux_gnueabi),
    ("armv7-unknown-linux-gnueabihf", armv7_unknown_linux_gnueabihf),
    ("armv7-unknown-linux-musleabihf", armv7_unknown_linux_musleabihf),
    ("aarch64-unknown-linux-gnu", aarch64_unknown_linux_gnu),
    ("x86_64-unknown-linux-musl", x86_64_unknown_linux_musl),
    ("i686-unknown-linux-musl", i686_unknown_linux_musl),
    ("mips-unknown-linux-musl", mips_unknown_linux_musl),
    ("mipsel-unknown-linux-musl", mipsel_unknown_linux_musl),
    ("mips-unknown-linux-uclibc", mips_unknown_linux_uclibc),
    ("mipsel-unknown-linux-uclibc", mipsel_unknown_linux_uclibc),

    ("sparc64-unknown-linux-gnu", sparc64_unknown_linux_gnu),

    ("i686-linux-android", i686_linux_android),
    ("arm-linux-androideabi", arm_linux_androideabi),
    ("armv7-linux-androideabi", armv7_linux_androideabi),
    ("aarch64-linux-android", aarch64_linux_android),

    ("i686-unknown-freebsd", i686_unknown_freebsd),
    ("x86_64-unknown-freebsd", x86_64_unknown_freebsd),

    ("i686-unknown-dragonfly", i686_unknown_dragonfly),
    ("x86_64-unknown-dragonfly", x86_64_unknown_dragonfly),

    ("x86_64-unknown-bitrig", x86_64_unknown_bitrig),

    ("i686-unknown-openbsd", i686_unknown_openbsd),
    ("x86_64-unknown-openbsd", x86_64_unknown_openbsd),

    ("sparc64-unknown-netbsd", sparc64_unknown_netbsd),
    ("x86_64-unknown-netbsd", x86_64_unknown_netbsd),
    ("x86_64-rumprun-netbsd", x86_64_rumprun_netbsd),

    ("i686-unknown-haiku", i686_unknown_haiku),
    ("x86_64-unknown-haiku", x86_64_unknown_haiku),

    ("x86_64-apple-darwin", x86_64_apple_darwin),
    ("i686-apple-darwin", i686_apple_darwin),

    ("aarch64-unknown-fuchsia", aarch64_unknown_fuchsia),
    ("x86_64-unknown-fuchsia", x86_64_unknown_fuchsia),

    ("x86_64-unknown-redox", x86_64_unknown_redox),

    ("i386-apple-ios", i386_apple_ios),
    ("x86_64-apple-ios", x86_64_apple_ios),
    ("aarch64-apple-ios", aarch64_apple_ios),
    ("armv7-apple-ios", armv7_apple_ios),
    ("armv7s-apple-ios", armv7s_apple_ios),

    ("x86_64-sun-solaris", x86_64_sun_solaris),

    ("x86_64-pc-windows-gnu", x86_64_pc_windows_gnu),
    ("i686-pc-windows-gnu", i686_pc_windows_gnu),

    ("x86_64-pc-windows-msvc", x86_64_pc_windows_msvc),
    ("i686-pc-windows-msvc", i686_pc_windows_msvc),
    ("i586-pc-windows-msvc", i586_pc_windows_msvc),

    ("le32-unknown-nacl", le32_unknown_nacl),
    ("asmjs-unknown-emscripten", asmjs_unknown_emscripten),
    ("wasm32-unknown-emscripten", wasm32_unknown_emscripten),

    ("thumbv6m-none-eabi", thumbv6m_none_eabi),
    ("thumbv7m-none-eabi", thumbv7m_none_eabi),
    ("thumbv7em-none-eabi", thumbv7em_none_eabi),
    ("thumbv7em-none-eabihf", thumbv7em_none_eabihf),
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
    /// Optional settings with defaults.
    pub options: TargetOptions,
}

/// Optional aspects of a target specification.
///
/// This has an implementation of `Default`, see each field for what the default is. In general,
/// these try to take "minimal defaults" that don't assume anything about the runtime they run in.
#[derive(PartialEq, Clone, Debug)]
pub struct TargetOptions {
    /// Whether the target is built-in or loaded from a custom target specification.
    pub is_builtin: bool,

    /// Linker to invoke. Defaults to "cc".
    pub linker: String,
    /// Archive utility to use when managing archives. Defaults to "ar".
    pub ar: String,

    /// Linker arguments that are unconditionally passed *before* any
    /// user-defined libraries.
    pub pre_link_args: Vec<String>,
    /// Objects to link before all others, always found within the
    /// sysroot folder.
    pub pre_link_objects_exe: Vec<String>, // ... when linking an executable
    pub pre_link_objects_dll: Vec<String>, // ... when linking a dylib
    /// Linker arguments that are unconditionally passed after any
    /// user-defined but before post_link_objects.  Standard platform
    /// libraries that should be always be linked to, usually go here.
    pub late_link_args: Vec<String>,
    /// Objects to link after all others, always found within the
    /// sysroot folder.
    pub post_link_objects: Vec<String>,
    /// Linker arguments that are unconditionally passed *after* any
    /// user-defined libraries.
    pub post_link_args: Vec<String>,

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
    /// Whether executables are available on this target. iOS, for example, only allows static
    /// libraries. Defaults to false.
    pub executables: bool,
    /// Relocation model to use in object file. Corresponds to `llc
    /// -relocation-model=$relocation_model`. Defaults to "pic".
    pub relocation_model: String,
    /// Code model to use. Corresponds to `llc -code-model=$code_model`. Defaults to "default".
    pub code_model: String,
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
    /// Whether the target toolchain is like OpenBSD's.
    /// Only useful for compiling against OpenBSD, for configuring abi when returning a struct.
    pub is_like_openbsd: bool,
    /// Whether the target toolchain is like OSX's. Only useful for compiling against iOS/OS X, in
    /// particular running dsymutil and some other stuff like `-dead_strip`. Defaults to false.
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
    /// Whether the linker support GNU-like arguments such as -O. Defaults to false.
    pub linker_is_gnu: bool,
    /// The MinGW toolchain has a known issue that prevents it from correctly
    /// handling COFF object files with more than 2^15 sections. Since each weak
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

    /// Default crate for allocation symbols to link against
    pub lib_allocation_crate: String,
    pub exe_allocation_crate: String,

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

    /// Panic strategy: "unwind" or "abort"
    pub panic_strategy: PanicStrategy,

    /// A blacklist of ABIs unsupported by the current target. Note that generic
    /// ABIs are considered to be supported on all platforms and cannot be blacklisted.
    pub abi_blacklist: Vec<Abi>,

    /// Whether or not the CRT is statically linked by default.
    pub crt_static_default: bool,
}

impl Default for TargetOptions {
    /// Create a set of "sane defaults" for any target. This is still
    /// incomplete, and if used for compilation, will certainly not work.
    fn default() -> TargetOptions {
        TargetOptions {
            is_builtin: false,
            linker: option_env!("CFG_DEFAULT_LINKER").unwrap_or("cc").to_string(),
            ar: option_env!("CFG_DEFAULT_AR").unwrap_or("ar").to_string(),
            pre_link_args: Vec::new(),
            post_link_args: Vec::new(),
            asm_args: Vec::new(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            dynamic_linking: false,
            executables: false,
            relocation_model: "pic".to_string(),
            code_model: "default".to_string(),
            disable_redzone: false,
            eliminate_frame_pointer: true,
            function_sections: true,
            dll_prefix: "lib".to_string(),
            dll_suffix: ".so".to_string(),
            exe_suffix: "".to_string(),
            staticlib_prefix: "lib".to_string(),
            staticlib_suffix: ".a".to_string(),
            target_family: None,
            is_like_openbsd: false,
            is_like_osx: false,
            is_like_solaris: false,
            is_like_windows: false,
            is_like_android: false,
            is_like_msvc: false,
            linker_is_gnu: false,
            allows_weak_linkage: true,
            has_rpath: false,
            no_default_libraries: true,
            position_independent_executables: false,
            pre_link_objects_exe: Vec::new(),
            pre_link_objects_dll: Vec::new(),
            post_link_objects: Vec::new(),
            late_link_args: Vec::new(),
            archive_format: "gnu".to_string(),
            custom_unwind_resume: false,
            lib_allocation_crate: "alloc_system".to_string(),
            exe_allocation_crate: "alloc_system".to_string(),
            allow_asm: true,
            has_elf_tls: false,
            obj_is_bitcode: false,
            no_integrated_as: false,
            min_atomic_width: None,
            max_atomic_width: None,
            panic_strategy: PanicStrategy::Unwind,
            abi_blacklist: vec![],
            crt_static_default: false,
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
        self.options.max_atomic_width.unwrap_or(self.target_pointer_width.parse().unwrap())
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
            match obj.find(name)
                     .map(|s| s.as_string())
                     .and_then(|os| os.map(|s| s.to_string())) {
                Some(val) => Ok(val),
                None => {
                    return Err(format!("Field {} in target specification is required", name))
                }
            }
        };

        let get_opt_field = |name: &str, default: &str| {
            obj.find(name).and_then(|s| s.as_string())
               .map(|s| s.to_string())
               .unwrap_or(default.to_string())
        };

        let mut base = Target {
            llvm_target: get_req_field("llvm-target")?,
            target_endian: get_req_field("target-endian")?,
            target_pointer_width: get_req_field("target-pointer-width")?,
            data_layout: get_req_field("data-layout")?,
            arch: get_req_field("arch")?,
            target_os: get_req_field("os")?,
            target_env: get_opt_field("env", ""),
            target_vendor: get_opt_field("vendor", "unknown"),
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
        }

        key!(is_builtin, bool);
        key!(linker);
        key!(ar);
        key!(pre_link_args, list);
        key!(pre_link_objects_exe, list);
        key!(pre_link_objects_dll, list);
        key!(late_link_args, list);
        key!(post_link_objects, list);
        key!(post_link_args, list);
        key!(asm_args, list);
        key!(cpu);
        key!(features);
        key!(dynamic_linking, bool);
        key!(executables, bool);
        key!(relocation_model);
        key!(code_model);
        key!(disable_redzone, bool);
        key!(eliminate_frame_pointer, bool);
        key!(function_sections, bool);
        key!(dll_prefix);
        key!(dll_suffix);
        key!(exe_suffix);
        key!(staticlib_prefix);
        key!(staticlib_suffix);
        key!(target_family, optional);
        key!(is_like_openbsd, bool);
        key!(is_like_osx, bool);
        key!(is_like_solaris, bool);
        key!(is_like_windows, bool);
        key!(is_like_msvc, bool);
        key!(is_like_android, bool);
        key!(linker_is_gnu, bool);
        key!(allows_weak_linkage, bool);
        key!(has_rpath, bool);
        key!(no_default_libraries, bool);
        key!(position_independent_executables, bool);
        key!(archive_format);
        key!(allow_asm, bool);
        key!(custom_unwind_resume, bool);
        key!(lib_allocation_crate);
        key!(exe_allocation_crate);
        key!(has_elf_tls, bool);
        key!(obj_is_bitcode, bool);
        key!(no_integrated_as, bool);
        key!(max_atomic_width, Option<u64>);
        key!(min_atomic_width, Option<u64>);
        try!(key!(panic_strategy, PanicStrategy));
        key!(crt_static_default, bool);

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
    pub fn search(target: &str) -> Result<Target, String> {
        use std::env;
        use std::ffi::OsString;
        use std::fs::File;
        use std::path::{Path, PathBuf};
        use serialize::json;

        fn load_file(path: &Path) -> Result<Target, String> {
            let mut f = File::open(path).map_err(|e| e.to_string())?;
            let mut contents = Vec::new();
            f.read_to_end(&mut contents).map_err(|e| e.to_string())?;
            let obj = json::from_reader(&mut &contents[..])
                           .map_err(|e| e.to_string())?;
            Target::from_json(obj)
        }

        if let Ok(t) = load_specific(target) {
            return Ok(t)
        }

        let path = Path::new(target);

        if path.is_file() {
            return load_file(&path);
        }

        let path = {
            let mut target = target.to_string();
            target.push_str(".json");
            PathBuf::from(target)
        };

        let target_path = env::var_os("RUST_TARGET_PATH")
                              .unwrap_or(OsString::new());

        // FIXME 16351: add a sane default search path?

        for dir in env::split_paths(&target_path) {
            let p =  dir.join(&path);
            if p.is_file() {
                return load_file(&p);
            }
        }

        Err(format!("Could not find specification for target {:?}", target))
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
        }

        target_val!(llvm_target);
        target_val!(target_endian);
        target_val!(target_pointer_width);
        target_val!(arch);
        target_val!(target_os, "os");
        target_val!(target_env, "env");
        target_val!(target_vendor, "vendor");
        target_val!(arch);
        target_val!(data_layout);

        target_option_val!(is_builtin);
        target_option_val!(linker);
        target_option_val!(ar);
        target_option_val!(pre_link_args);
        target_option_val!(pre_link_objects_exe);
        target_option_val!(pre_link_objects_dll);
        target_option_val!(late_link_args);
        target_option_val!(post_link_objects);
        target_option_val!(post_link_args);
        target_option_val!(asm_args);
        target_option_val!(cpu);
        target_option_val!(features);
        target_option_val!(dynamic_linking);
        target_option_val!(executables);
        target_option_val!(relocation_model);
        target_option_val!(code_model);
        target_option_val!(disable_redzone);
        target_option_val!(eliminate_frame_pointer);
        target_option_val!(function_sections);
        target_option_val!(dll_prefix);
        target_option_val!(dll_suffix);
        target_option_val!(exe_suffix);
        target_option_val!(staticlib_prefix);
        target_option_val!(staticlib_suffix);
        target_option_val!(target_family);
        target_option_val!(is_like_openbsd);
        target_option_val!(is_like_osx);
        target_option_val!(is_like_solaris);
        target_option_val!(is_like_windows);
        target_option_val!(is_like_msvc);
        target_option_val!(is_like_android);
        target_option_val!(linker_is_gnu);
        target_option_val!(allows_weak_linkage);
        target_option_val!(has_rpath);
        target_option_val!(no_default_libraries);
        target_option_val!(position_independent_executables);
        target_option_val!(archive_format);
        target_option_val!(allow_asm);
        target_option_val!(custom_unwind_resume);
        target_option_val!(lib_allocation_crate);
        target_option_val!(exe_allocation_crate);
        target_option_val!(has_elf_tls);
        target_option_val!(obj_is_bitcode);
        target_option_val!(no_integrated_as);
        target_option_val!(min_atomic_width);
        target_option_val!(max_atomic_width);
        target_option_val!(panic_strategy);
        target_option_val!(crt_static_default);

        if default.abi_blacklist != self.options.abi_blacklist {
            d.insert("abi-blacklist".to_string(), self.options.abi_blacklist.iter()
                .map(Abi::name).map(|name| name.to_json())
                .collect::<Vec<_>>().to_json());
        }

        Json::Object(d)
    }
}

fn maybe_jemalloc() -> String {
    if cfg!(feature = "jemalloc") {
        "alloc_jemalloc".to_string()
    } else {
        "alloc_system".to_string()
    }
}
