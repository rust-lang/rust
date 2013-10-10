// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Context data structure used by rustpkg

use std::{io, os};
use extra::workcache;
use rustc::driver::session::{OptLevel, No};

#[deriving(Clone)]
pub struct Context {
    // Config strings that the user passed in with --cfg
    cfgs: ~[~str],
    // Flags to pass to rustc
    rustc_flags: RustcFlags,
    // If use_rust_path_hack is true, rustpkg searches for sources
    // in *package* directories that are in the RUST_PATH (for example,
    // FOO/src/bar-0.1 instead of FOO). The flag doesn't affect where
    // rustpkg stores build artifacts.
    use_rust_path_hack: bool,
    // The root directory containing the Rust standard libraries
    sysroot: Path
}

#[deriving(Clone)]
pub struct BuildContext {
    // Context for workcache
    workcache_context: workcache::Context,
    // Everything else
    context: Context
}

impl BuildContext {
    pub fn sysroot(&self) -> Path {
        self.context.sysroot.clone()
    }

    pub fn sysroot_to_use(&self) -> Path {
        self.context.sysroot_to_use()
    }

    /// Returns the flags to pass to rustc, as a vector of strings
    pub fn flag_strs(&self) -> ~[~str] {
        self.context.flag_strs()
    }

    pub fn compile_upto(&self) -> StopBefore {
        self.context.compile_upto()
    }
}

/*
Deliberately unsupported rustc flags:
   --bin, --lib           inferred from crate file names
   -L                     inferred from extern mods
   --out-dir              inferred from RUST_PATH
   --test                 use `rustpkg test`
   -v -h --ls             don't make sense with rustpkg
   -W -A -D -F -          use pragmas instead

rustc flags that aren't implemented yet:
   --passes
   --llvm-arg
   --target-feature
   --android-cross-path
*/
pub struct RustcFlags {
    compile_upto: StopBefore,
    // Linker to use with the --linker flag
    linker: Option<~str>,
    // Extra arguments to pass to rustc with the --link-args flag
    link_args: Option<~str>,
    // Optimization level. 0 = default. -O = 2.
    optimization_level: OptLevel,
    // True if the user passed in --save-temps
    save_temps: bool,
    // Target (defaults to rustc's default target)
    target: Option<~str>,
    // Target CPU (defaults to rustc's default target CPU)
    target_cpu: Option<~str>,
    // Any -Z features
    experimental_features: Option<~[~str]>
}

impl Clone for RustcFlags {
    fn clone(&self) -> RustcFlags {
        RustcFlags {
            compile_upto: self.compile_upto,
            linker: self.linker.clone(),
            link_args: self.link_args.clone(),
            optimization_level: self.optimization_level,
            save_temps: self.save_temps,
            target: self.target.clone(),
            target_cpu: self.target_cpu.clone(),
            experimental_features: self.experimental_features.clone()
        }
    }
}

#[deriving(Eq)]
pub enum StopBefore {
    Nothing,  // compile everything
    Link,     // --no-link
    LLVMCompileBitcode, // --emit-llvm without -S
    LLVMAssemble, // -S --emit-llvm
    Assemble, // -S without --emit-llvm
    Trans,    // --no-trans
    Pretty,   // --pretty
    Analysis, // --parse-only
}

impl Context {
    pub fn sysroot(&self) -> Path {
        self.sysroot.clone()
    }

    /// Debugging
    pub fn sysroot_str(&self) -> ~str {
        self.sysroot.to_str()
    }

    // Hack so that rustpkg can run either out of a rustc target dir,
    // or the host dir
    pub fn sysroot_to_use(&self) -> Path {
        if !in_target(&self.sysroot) {
            self.sysroot.clone()
        } else {
            self.sysroot.pop().pop().pop()
        }
    }

    /// Returns the flags to pass to rustc, as a vector of strings
    pub fn flag_strs(&self) -> ~[~str] {
        self.rustc_flags.flag_strs()
    }

    pub fn compile_upto(&self) -> StopBefore {
        self.rustc_flags.compile_upto
    }
}

/// We assume that if ../../rustc exists, then we're running
/// rustpkg from a Rust target directory. This is part of a
/// kludgy hack used to adjust the sysroot.
pub fn in_target(sysroot: &Path) -> bool {
    debug2!("Checking whether {} is in target", sysroot.to_str());
    os::path_is_dir(&sysroot.pop().pop().push("rustc"))
}

impl RustcFlags {
    fn flag_strs(&self) -> ~[~str] {
        let linker_flag = match self.linker {
            Some(ref l) => ~[~"--linker", l.clone()],
            None    => ~[]
        };
        let link_args_flag = match self.link_args {
            Some(ref l) => ~[~"--link-args", l.clone()],
            None        => ~[]
        };
        let save_temps_flag = if self.save_temps { ~[~"--save-temps"] } else { ~[] };
        let target_flag = match self.target {
            Some(ref l) => ~[~"--target", l.clone()],
            None        => ~[]
        };
        let target_cpu_flag = match self.target_cpu {
            Some(ref l) => ~[~"--target-cpu", l.clone()],
            None        => ~[]
        };
        let z_flags = match self.experimental_features {
            Some(ref ls)    => ls.flat_map(|s| ~[~"-Z", s.clone()]),
            None            => ~[]
        };
        linker_flag
            + link_args_flag
            + save_temps_flag
            + target_flag
            + target_cpu_flag
            + z_flags + (match self.compile_upto {
            LLVMCompileBitcode => ~[~"--emit-llvm"],
            LLVMAssemble => ~[~"--emit-llvm", ~"-S"],
            Link => ~[~"-c"],
            Trans => ~[~"--no-trans"],
            Assemble => ~[~"-S"],
            // n.b. Doesn't support all flavors of --pretty (yet)
            Pretty => ~[~"--pretty"],
            Analysis => ~[~"--parse-only"],
            Nothing => ~[]
        })
    }

    pub fn default() -> RustcFlags {
        RustcFlags {
            linker: None,
            link_args: None,
            compile_upto: Nothing,
            optimization_level: No,
            save_temps: false,
            target: None,
            target_cpu: None,
            experimental_features: None
        }
    }
}

/// Returns true if any of the flags given are incompatible with the cmd
pub fn flags_forbidden_for_cmd(flags: &RustcFlags,
                        cfgs: &[~str],
                        cmd: &str, user_supplied_opt_level: bool) -> bool {
    let complain = |s| {
        println!("The {} option can only be used with the `build` command:
                  rustpkg [options..] build {} [package-ID]", s, s);
    };

    if flags.linker.is_some() && cmd != "build" && cmd != "install" {
        io::println("The --linker option can only be used with the build or install commands.");
        return true;
    }
    if flags.link_args.is_some() && cmd != "build" && cmd != "install" {
        io::println("The --link-args option can only be used with the build or install commands.");
        return true;
    }

    if !cfgs.is_empty() && cmd != "build" && cmd != "install" {
        io::println("The --cfg option can only be used with the build or install commands.");
        return true;
    }

    if user_supplied_opt_level && cmd != "build" && cmd != "install" {
        io::println("The -O and --opt-level options can only be used with the build \
                    or install commands.");
        return true;
    }

    if flags.save_temps  && cmd != "build" && cmd != "install" {
        io::println("The --save-temps option can only be used with the build \
                    or install commands.");
        return true;
    }

    if flags.target.is_some()  && cmd != "build" && cmd != "install" {
        io::println("The --target option can only be used with the build \
                    or install commands.");
        return true;
    }
    if flags.target_cpu.is_some()  && cmd != "build" && cmd != "install" {
        io::println("The --target-cpu option can only be used with the build \
                    or install commands.");
        return true;
    }
    if flags.experimental_features.is_some() && cmd != "build" && cmd != "install" {
        io::println("The -Z option can only be used with the build or install commands.");
        return true;
    }

    match flags.compile_upto {
        Link if cmd != "build" => {
            complain("--no-link");
            true
        }
        Trans if cmd != "build" => {
            complain("--no-trans");
            true
        }
        Assemble if cmd != "build" => {
            complain("-S");
            true
        }
        Pretty if cmd != "build" => {
            complain("--pretty");
            true
        }
        Analysis if cmd != "build" => {
            complain("--parse-only");
            true
        }
        LLVMCompileBitcode if cmd != "build" => {
            complain("--emit-llvm");
            true
        }
        LLVMAssemble if cmd != "build" => {
            complain("--emit-llvm");
            true
        }
        _ => false
    }
}
