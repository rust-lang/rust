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
//! will be given.  `RUST_TARGET_PATH` includes `/etc/rustc` as its last entry,
//! to be searched by default.
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
//! `data-layout`, `llvm-target`, `target-endian`, `target-pointer-width`, and
//! `arch`. In general, options passed to rustc with `-C` override the target's
//! settings, though `target-feature` and `link-args` will *add* to the list
//! specified by the target, rather than replace.

use serialize::json::Json;
use syntax::{diagnostic, abi};
use std::default::Default;
use std::old_io::fs::PathExtensions;

mod windows_base;
mod linux_base;
mod apple_base;
mod apple_ios_base;
mod freebsd_base;
mod dragonfly_base;
mod openbsd_base;

mod armv7_apple_ios;
mod armv7s_apple_ios;
mod i386_apple_ios;

mod arm_linux_androideabi;
mod arm_unknown_linux_gnueabi;
mod arm_unknown_linux_gnueabihf;
mod aarch64_apple_ios;
mod aarch64_linux_android;
mod aarch64_unknown_linux_gnu;
mod i686_apple_darwin;
mod i686_pc_windows_gnu;
mod i686_unknown_dragonfly;
mod i686_unknown_linux_gnu;
mod mips_unknown_linux_gnu;
mod mipsel_unknown_linux_gnu;
mod powerpc_unknown_linux_gnu;
mod x86_64_apple_darwin;
mod x86_64_apple_ios;
mod x86_64_pc_windows_gnu;
mod x86_64_unknown_freebsd;
mod x86_64_unknown_dragonfly;
mod x86_64_unknown_linux_gnu;
mod x86_64_unknown_openbsd;

/// Everything `rustc` knows about how to compile for a specific target.
///
/// Every field here must be specified, and has no default value.
#[derive(Clone, Debug)]
pub struct Target {
    /// [Data layout](http://llvm.org/docs/LangRef.html#data-layout) to pass to LLVM.
    pub data_layout: String,
    /// Target triple to pass to LLVM.
    pub llvm_target: String,
    /// String to use as the `target_endian` `cfg` variable.
    pub target_endian: String,
    /// String to use as the `target_pointer_width` `cfg` variable.
    pub target_pointer_width: String,
    /// OS name to use for conditional compilation.
    pub target_os: String,
    /// Architecture to use for ABI considerations. Valid options: "x86", "x86_64", "arm",
    /// "aarch64", "mips", and "powerpc". "mips" includes "mipsel".
    pub arch: String,
    /// Optional settings with defaults.
    pub options: TargetOptions,
}

/// Optional aspects of a target specification.
///
/// This has an implementation of `Default`, see each field for what the default is. In general,
/// these try to take "minimal defaults" that don't assume anything about the runtime they run in.
#[derive(Clone, Debug)]
pub struct TargetOptions {
    /// Linker to invoke. Defaults to "cc".
    pub linker: String,
    /// Linker arguments that are unconditionally passed *before* any user-defined libraries.
    pub pre_link_args: Vec<String>,
    /// Linker arguments that are unconditionally passed *after* any user-defined libraries.
    pub post_link_args: Vec<String>,
    /// Default CPU to pass to LLVM. Corresponds to `llc -mcpu=$cpu`. Defaults to "default".
    pub cpu: String,
    /// Default target features to pass to LLVM. These features will *always* be passed, and cannot
    /// be disabled even via `-C`. Corresponds to `llc -mattr=$features`.
    pub features: String,
    /// Whether dynamic linking is available on this target. Defaults to false.
    pub dynamic_linking: bool,
    /// Whether executables are available on this target. iOS, for example, only allows static
    /// libraries. Defaults to false.
    pub executables: bool,
    /// Whether LLVM's segmented stack prelude is supported by whatever runtime is available.
    /// Will emit stack checks and calls to __morestack. Defaults to false.
    pub morestack: bool,
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
    /// Whether the target toolchain is like OSX's. Only useful for compiling against iOS/OS X, in
    /// particular running dsymutil and some other stuff like `-dead_strip`. Defaults to false.
    pub is_like_osx: bool,
    /// Whether the target toolchain is like Windows'. Only useful for compiling against Windows,
    /// only realy used for figuring out how to find libraries, since Windows uses its own
    /// library naming convention. Defaults to false.
    pub is_like_windows: bool,
    /// Whether the target toolchain is like Android's. Only useful for compiling against Android.
    /// Defaults to false.
    pub is_like_android: bool,
    /// Whether the linker support GNU-like arguments such as -O. Defaults to false.
    pub linker_is_gnu: bool,
    /// Whether the linker support rpaths or not. Defaults to false.
    pub has_rpath: bool,
    /// Whether to disable linking to compiler-rt. Defaults to false, as LLVM will emit references
    /// to the functions that compiler-rt provides.
    pub no_compiler_rt: bool,
    /// Dynamically linked executables can be compiled as position independent if the default
    /// relocation model of position independent code is not changed. This is a requirement to take
    /// advantage of ASLR, as otherwise the functions in the executable are not randomized and can
    /// be used during an exploit of a vulnerability in any code.
    pub position_independent_executables: bool,
}

impl Default for TargetOptions {
    /// Create a set of "sane defaults" for any target. This is still incomplete, and if used for
    /// compilation, will certainly not work.
    fn default() -> TargetOptions {
        TargetOptions {
            linker: "cc".to_string(),
            pre_link_args: Vec::new(),
            post_link_args: Vec::new(),
            cpu: "generic".to_string(),
            features: "".to_string(),
            dynamic_linking: false,
            executables: false,
            morestack: false,
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
            is_like_osx: false,
            is_like_windows: false,
            is_like_android: false,
            linker_is_gnu: false,
            has_rpath: false,
            no_compiler_rt: false,
            position_independent_executables: false,
        }
    }
}

impl Target {
    /// Given a function ABI, turn "System" into the correct ABI for this target.
    pub fn adjust_abi(&self, abi: abi::Abi) -> abi::Abi {
        match abi {
            abi::System => {
                if self.options.is_like_windows && self.arch == "x86" {
                    abi::Stdcall
                } else {
                    abi::C
                }
            },
            abi => abi
        }
    }

    /// Load a target descriptor from a JSON object.
    pub fn from_json(obj: Json) -> Target {
        // this is 1. ugly, 2. error prone.


        let handler = diagnostic::default_handler(diagnostic::Auto, None, true);

        let get_req_field = |name: &str| {
            match obj.find(name)
                     .map(|s| s.as_string())
                     .and_then(|os| os.map(|s| s.to_string())) {
                Some(val) => val,
                None =>
                    handler.fatal(&format!("Field {} in target specification is required", name)[])
            }
        };

        let mut base = Target {
            data_layout: get_req_field("data-layout"),
            llvm_target: get_req_field("llvm-target"),
            target_endian: get_req_field("target-endian"),
            target_pointer_width: get_req_field("target-pointer-width"),
            arch: get_req_field("arch"),
            target_os: get_req_field("os"),
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
            ($key_name:ident, list) => ( {
                let name = (stringify!($key_name)).replace("_", "-");
                obj.find(&name[..]).map(|o| o.as_array()
                    .map(|v| base.options.$key_name = v.iter()
                        .map(|a| a.as_string().unwrap().to_string()).collect()
                        )
                    );
            } );
        }

        key!(cpu);
        key!(linker);
        key!(relocation_model);
        key!(code_model);
        key!(dll_prefix);
        key!(dll_suffix);
        key!(exe_suffix);
        key!(staticlib_prefix);
        key!(staticlib_suffix);
        key!(features);
        key!(dynamic_linking, bool);
        key!(executables, bool);
        key!(morestack, bool);
        key!(disable_redzone, bool);
        key!(eliminate_frame_pointer, bool);
        key!(function_sections, bool);
        key!(is_like_osx, bool);
        key!(is_like_windows, bool);
        key!(linker_is_gnu, bool);
        key!(has_rpath, bool);
        key!(no_compiler_rt, bool);
        key!(pre_link_args, list);
        key!(post_link_args, list);

        base
    }

    /// Search RUST_TARGET_PATH for a JSON file specifying the given target triple. Note that it
    /// could also just be a bare filename already, so also check for that. If one of the hardcoded
    /// targets we know about, just return it directly.
    ///
    /// The error string could come from any of the APIs called, including filesystem access and
    /// JSON decoding.
    pub fn search(target: &str) -> Result<Target, String> {
        use std::env;
        use std::ffi::OsString;
        use std::old_io::File;
        use std::old_path::Path;
        use serialize::json;

        fn load_file(path: &Path) -> Result<Target, String> {
            let mut f = try!(File::open(path).map_err(|e| format!("{:?}", e)));
            let obj = try!(json::from_reader(&mut f).map_err(|e| format!("{:?}", e)));
            Ok(Target::from_json(obj))
        }

        // this would use a match if stringify! were allowed in pattern position
        macro_rules! load_specific {
            ( $($name:ident),+ ) => (
                {
                    let target = target.replace("-", "_");
                    if false { }
                    $(
                        else if target == stringify!($name) {
                            let t = $name::target();
                            debug!("Got builtin target: {:?}", t);
                            return Ok(t);
                        }
                    )*
                    else if target == "x86_64-w64-mingw32" {
                        let t = x86_64_pc_windows_gnu::target();
                        return Ok(t);
                    } else if target == "i686-w64-mingw32" {
                        let t = i686_pc_windows_gnu::target();
                        return Ok(t);
                    }
                }
            )
        }

        load_specific!(
            x86_64_unknown_linux_gnu,
            i686_unknown_linux_gnu,
            mips_unknown_linux_gnu,
            mipsel_unknown_linux_gnu,
            powerpc_unknown_linux_gnu,
            arm_unknown_linux_gnueabi,
            arm_unknown_linux_gnueabihf,
            aarch64_unknown_linux_gnu,

            arm_linux_androideabi,
            aarch64_linux_android,

            x86_64_unknown_freebsd,

            i686_unknown_dragonfly,
            x86_64_unknown_dragonfly,

            x86_64_unknown_openbsd,

            x86_64_apple_darwin,
            i686_apple_darwin,

            i386_apple_ios,
            x86_64_apple_ios,
            aarch64_apple_ios,
            armv7_apple_ios,
            armv7s_apple_ios,

            x86_64_pc_windows_gnu,
            i686_pc_windows_gnu
        );


        let path = Path::new(target);

        if path.is_file() {
            return load_file(&path);
        }

        let path = {
            let mut target = target.to_string();
            target.push_str(".json");
            Path::new(target)
        };

        let target_path = env::var_os("RUST_TARGET_PATH").unwrap_or(OsString::from_str(""));

        // FIXME 16351: add a sane default search path?

        for dir in env::split_paths(&target_path) {
            let p =  dir.join(path.clone());
            if p.is_file() {
                return load_file(&p);
            }
        }

        Err(format!("Could not find specification for target {:?}", target))
    }
}
