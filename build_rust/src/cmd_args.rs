// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Use `getopts` to parse the command line arguments. Note this
//! module does minimal processing and it is up to `mod configure`
//! to interpret the command line arguments and set the configure
//! variables accordingly.

extern crate getopts;

use build_state::*;

pub struct CmdArgs {
    pub host_triple : Option<String>,
    pub target_triples : Vec<String>,
    pub build_dir : Option<String>,
    pub rustc_root_dir : Option<String>,
    pub nproc : Option<u8>,
    pub no_clean_build : bool,
    pub use_local_rustc : bool,
    pub no_reconfigure_llvm : bool,
    pub no_rebuild_llvm : bool,
    pub no_bootstrap : bool,
    pub enable_debug_build : bool,
    pub enable_llvm_assertions : bool,
    pub disable_llvm_assertions : bool,
    pub verbose : bool
}

fn get_all_options() -> getopts::Options {
    let mut opts = getopts::Options::new();
    opts.optflag("h", "help", "Print this help menu.");
    opts.optopt("", "build-dir",
                "Specify the build directory where all build artifacts are written into.",
                "DIR");
    opts.optopt("", "rustc-root",
                "Specify the root directory for the rustc source code.",
                "DIR");
    opts.optopt("", "host",
                "Specify the host triple, ie. the triple on which the compiler runs.",
                "HOST");
    opts.optopt("", "target",
                "Specify the tearget triples that the compiler can generate code for.",
                "TARGET1[,TARGET2,...]");
    opts.optopt("", "nproc",
                "Specify the number of parallel jobs the build can use. Note that for MSVC builds, this parameter is ignored and all available cores are used.",
                "NUM");
    opts.optflag("", "no-clean-build",
                 "Do not clean the build directories.");
    opts.optflag("", "use-local-rustc",
                 "Do not download the snapshot. Use local rustc to bootstrap stage0 build.");
    opts.optflag("", "no-reconfigure-llvm",
                 "Do not re-configure llvm. Build will fail if llvm has not been configured previously. This implies --no-clean-build.");
    opts.optflag("", "no-rebuild-llvm",
                 "Do not rebuild llvm. Build will fail if llvm has not been built previously. This implies --no-clean-build.");
    opts.optflag("", "no-bootstrap",
                 "Do not bootstrap. Build stage2 binaries only. Build will fail if not already bootstrapped.");
    opts.optflag("", "enable-debug-build",
                 "Build the compiler in debug mode. The default is a release build with optimizations on.");
    opts.optflag("", "enable-llvm-assertions",
                 "Build LLVM with assertions on. This is implied by --enable-debug-build.");
    opts.optflag("", "disable-llvm-assertions",
                 "Build LLVM with assertions off. This is the default when building in release mode.");
    opts.optflag("v", "verbose",
                 "Show extra information during build.");
    opts
}

fn get_usage_string(prog : &str, opts: &getopts::Options) -> String {
    let brief = format!("Usage: {}", prog);
    opts.usage(&brief)
}

pub fn parse_cmd_args() -> BuildState<CmdArgs> {
    let opts = get_all_options();
    let args : Vec<String> = ::std::env::args().collect();
    let program = &args[0];
    let matches = try!(opts.parse(&args[1..]));
    if matches.opt_present("h") {
        return msg_stop(get_usage_string(program, &opts));
    }

    let cmd_args = CmdArgs {
        host_triple : matches.opt_str("host"),
        target_triples : matches.opt_strs("target"),
        build_dir : matches.opt_str("build-dir"),
        rustc_root_dir : matches.opt_str("rustc-root"),
        nproc : matches.opt_str("nproc").and_then(|s| s.parse::<u8>().ok()),
        no_clean_build : matches.opt_present("no-clean-build"),
        use_local_rustc : matches.opt_present("use-local-rustc"),
        no_reconfigure_llvm : matches.opt_present("no-reconfigure-llvm"),
        no_rebuild_llvm : matches.opt_present("no-rebuild-llvm"),
        no_bootstrap : matches.opt_present("no-bootstrap"),
        enable_debug_build : matches.opt_present("enable-debug-build"),
        enable_llvm_assertions : matches.opt_present("enable-llvm-assertions"),
        disable_llvm_assertions : matches.opt_present(
            "disable-llvm-assertions"),
        verbose : matches.opt_present("verbose")
    };

    continue_with(cmd_args)
}
