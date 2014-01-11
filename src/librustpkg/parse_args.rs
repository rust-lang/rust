// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use context::{Context, RustcFlags, Trans, Link, Nothing, Pretty, Analysis, Assemble,
                       LLVMAssemble, LLVMCompileBitcode};
use context::{Command, flags_forbidden_for_cmd};
use rustc::version;
use exit_codes::{BAD_FLAG_CODE};
use rustc::driver::{session};

use usage;

use extra::{getopts};
use std::{result};
use std::hashmap::HashSet;

///  Result of parsing command line arguments
pub struct ParseResult {
    // Command
    command: Command,
    // Args
    args: ~[~str],
    // Parsed command line flags
    context: Context,
    // Path to system root
    sysroot: Option<~str>
}

/// Parses command line arguments of rustpkg.
/// Returns a triplet (command, remaining_args, context)
pub fn parse_args(args: &[~str]) -> Result<ParseResult, int> {
    let opts = ~[ getopts::optflag("h"), getopts::optflag("help"),
                                        getopts::optflag("no-link"),
                                        getopts::optflag("no-trans"),
                 // n.b. Ignores different --pretty options for now
                                        getopts::optflag("pretty"),
                                        getopts::optflag("parse-only"),
                 getopts::optflag("S"), getopts::optflag("assembly"),
                 getopts::optmulti("c"), getopts::optmulti("cfg"),
                 getopts::optflag("v"), getopts::optflag("version"),
                 getopts::optflag("r"), getopts::optflag("rust-path-hack"),
                                        getopts::optopt("sysroot"),
                                        getopts::optflag("emit-llvm"),
                                        getopts::optopt("linker"),
                                        getopts::optopt("link-args"),
                                        getopts::optopt("opt-level"),
                 getopts::optflag("O"),
                                        getopts::optflag("save-temps"),
                                        getopts::optopt("target"),
                                        getopts::optopt("target-cpu"),
                 getopts::optmulti("Z")                                   ];
    let matches = &match getopts::getopts(args, opts) {
        result::Ok(m) => m,
        result::Err(f) => {
            error!("{}", f.to_err_msg());
            return Err(1);
        }
    };
    let no_link = matches.opt_present("no-link");
    let no_trans = matches.opt_present("no-trans");
    let supplied_sysroot = matches.opt_str("sysroot");
    let generate_asm = matches.opt_present("S") ||
        matches.opt_present("assembly");
    let parse_only = matches.opt_present("parse-only");
    let pretty = matches.opt_present("pretty");
    let emit_llvm = matches.opt_present("emit-llvm");

    if matches.opt_present("h") ||
       matches.opt_present("help") {
         usage::general();
         return Err(0);
    }

    if matches.opt_present("v") ||
       matches.opt_present("version") {
        version(args[0]);
        return Err(0);
    }

    let use_rust_path_hack = matches.opt_present("r") ||
                             matches.opt_present("rust-path-hack");

    let linker = matches.opt_str("linker");
    let link_args = matches.opt_str("link-args");
    let cfgs = matches.opt_strs("cfg") + matches.opt_strs("c");
    let mut user_supplied_opt_level = true;
    let opt_level = match matches.opt_str("opt-level") {
        Some(~"0") => session::No,
        Some(~"1") => session::Less,
        Some(~"2") => session::Default,
        Some(~"3") => session::Aggressive,
        _ if matches.opt_present("O") => session::Default,
        _ => {
            user_supplied_opt_level = false;
            session::No
        }
    };

    let save_temps = matches.opt_present("save-temps");
    let target     = matches.opt_str("target");
    let target_cpu = matches.opt_str("target-cpu");
    let experimental_features = {
        let strs = matches.opt_strs("Z");
        if matches.opt_present("Z") {
            Some(strs)
        }
        else {
            None
        }
    };

    let mut args = matches.free.clone();
    args.shift();

    if (args.len() < 1) {
        usage::general();
        return Err(1);
    }

    let rustc_flags = RustcFlags {
        linker: linker,
        link_args: link_args,
        optimization_level: opt_level,
        compile_upto: if no_trans {
            Trans
        } else if no_link {
            Link
        } else if pretty {
            Pretty
        } else if parse_only {
            Analysis
        } else if emit_llvm && generate_asm {
            LLVMAssemble
        } else if generate_asm {
            Assemble
        } else if emit_llvm {
            LLVMCompileBitcode
        } else {
            Nothing
        },
        save_temps: save_temps,
        target: target,
        target_cpu: target_cpu,
        additional_library_paths:
            HashSet::new(), // No way to set this from the rustpkg command line
        experimental_features: experimental_features
    };

    let cmd_opt = args.iter().filter_map( |s| from_str(s.clone())).next();
    let command = match(cmd_opt){
        None => {
            debug!("No legal command. Returning 0");
            usage::general();
            return Err(0);
        }
        Some(cmd) => {
            let bad_option = flags_forbidden_for_cmd(&rustc_flags,
                                                              cfgs,
                                                              cmd,
                                                              user_supplied_opt_level);
            if bad_option {
                usage::usage_for_command(cmd);
                debug!("Bad  option, returning BAD_FLAG_CODE");
                return Err(BAD_FLAG_CODE);
            } else {
                cmd
            }
        }
    };

    // Pop off all flags, plus the command
    let mut remaining_args: ~[~str] = args.iter().skip_while(|&s| {
        let maybe_command: Option<Command> = from_str(*s);
        maybe_command.is_none()
    }).map(|s| s.clone()).collect();
    remaining_args.shift();

    let context = Context{
        rustc_flags: rustc_flags,
        cfgs: cfgs,
        use_rust_path_hack: use_rust_path_hack,
    };
    Ok(ParseResult {
        command:  command,
        args: remaining_args,
        context: context,
        sysroot: supplied_sysroot
    })
}

