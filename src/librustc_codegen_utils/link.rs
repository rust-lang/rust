// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::io;
use std::iter;
use std::process::{Stdio, Output};

use cc::windows_registry;

use rustc::session::config::{self, OutputFilenames, Input, OutputType};
use rustc::session::Session;
use rustc::session::search_paths::PathKind;
use rustc_target::spec::LinkerFlavor;
use syntax::{ast, attr};
use syntax_pos::Span;

use command::Command;

pub fn out_filename(sess: &Session,
                crate_type: config::CrateType,
                outputs: &OutputFilenames,
                crate_name: &str)
                -> PathBuf {
    let default_filename = filename_for_input(sess, crate_type, crate_name, outputs);
    let out_filename = outputs.outputs.get(&OutputType::Exe)
                              .and_then(|s| s.to_owned())
                              .or_else(|| outputs.single_output_file.clone())
                              .unwrap_or(default_filename);

    check_file_is_writeable(&out_filename, sess);

    out_filename
}

// Make sure files are writeable.  Mac, FreeBSD, and Windows system linkers
// check this already -- however, the Linux linker will happily overwrite a
// read-only file.  We should be consistent.
pub fn check_file_is_writeable(file: &Path, sess: &Session) {
    if !is_writeable(file) {
        sess.fatal(&format!("output file {} is not writeable -- check its \
                            permissions", file.display()));
    }
}

fn is_writeable(p: &Path) -> bool {
    match p.metadata() {
        Err(..) => true,
        Ok(m) => !m.permissions().readonly()
    }
}

pub fn find_crate_name(sess: Option<&Session>,
                       attrs: &[ast::Attribute],
                       input: &Input) -> String {
    let validate = |s: String, span: Option<Span>| {
        ::rustc_metadata::validate_crate_name(sess, &s, span);
        s
    };

    // Look in attributes 100% of the time to make sure the attribute is marked
    // as used. After doing this, however, we still prioritize a crate name from
    // the command line over one found in the #[crate_name] attribute. If we
    // find both we ensure that they're the same later on as well.
    let attr_crate_name = attr::find_by_name(attrs, "crate_name")
        .and_then(|at| at.value_str().map(|s| (at, s)));

    if let Some(sess) = sess {
        if let Some(ref s) = sess.opts.crate_name {
            if let Some((attr, name)) = attr_crate_name {
                if name != &**s {
                    let msg = format!("--crate-name and #[crate_name] are \
                                       required to match, but `{}` != `{}`",
                                      s, name);
                    sess.span_err(attr.span, &msg);
                }
            }
            return validate(s.clone(), None);
        }
    }

    if let Some((attr, s)) = attr_crate_name {
        return validate(s.to_string(), Some(attr.span));
    }
    if let Input::File(ref path) = *input {
        if let Some(s) = path.file_stem().and_then(|s| s.to_str()) {
            if s.starts_with("-") {
                let msg = format!("crate names cannot start with a `-`, but \
                                   `{}` has a leading hyphen", s);
                if let Some(sess) = sess {
                    sess.err(&msg);
                }
            } else {
                return validate(s.replace("-", "_"), None);
            }
        }
    }

    "rust_out".to_string()
}

pub fn filename_for_metadata(sess: &Session,
                             crate_name: &str,
                             outputs: &OutputFilenames) -> PathBuf {
    let libname = format!("{}{}", crate_name, sess.opts.cg.extra_filename);

    let out_filename = outputs.single_output_file.clone()
        .unwrap_or_else(|| outputs.out_directory.join(&format!("lib{}.rmeta", libname)));

    check_file_is_writeable(&out_filename, sess);

    out_filename
}

pub fn filename_for_input(sess: &Session,
                          crate_type: config::CrateType,
                          crate_name: &str,
                          outputs: &OutputFilenames) -> PathBuf {
    let libname = format!("{}{}", crate_name, sess.opts.cg.extra_filename);

    match crate_type {
        config::CrateType::Rlib => {
            outputs.out_directory.join(&format!("lib{}.rlib", libname))
        }
        config::CrateType::Cdylib |
        config::CrateType::ProcMacro |
        config::CrateType::Dylib => {
            let (prefix, suffix) = (&sess.target.target.options.dll_prefix,
                                    &sess.target.target.options.dll_suffix);
            outputs.out_directory.join(&format!("{}{}{}", prefix, libname,
                                                suffix))
        }
        config::CrateType::Staticlib => {
            let (prefix, suffix) = (&sess.target.target.options.staticlib_prefix,
                                    &sess.target.target.options.staticlib_suffix);
            outputs.out_directory.join(&format!("{}{}{}", prefix, libname,
                                                suffix))
        }
        config::CrateType::Executable => {
            let suffix = &sess.target.target.options.exe_suffix;
            let out_filename = outputs.path(OutputType::Exe);
            if suffix.is_empty() {
                out_filename
            } else {
                out_filename.with_extension(&suffix[1..])
            }
        }
    }
}

/// Returns default crate type for target
///
/// Default crate type is used when crate type isn't provided neither
/// through cmd line arguments nor through crate attributes
///
/// It is CrateType::Executable for all platforms but iOS as there is no
/// way to run iOS binaries anyway without jailbreaking and
/// interaction with Rust code through static library is the only
/// option for now
pub fn default_output_for_target(sess: &Session) -> config::CrateType {
    if !sess.target.target.options.executables {
        config::CrateType::Staticlib
    } else {
        config::CrateType::Executable
    }
}

/// Checks if target supports crate_type as output
pub fn invalid_output_for_target(sess: &Session,
                                 crate_type: config::CrateType) -> bool {
    match crate_type {
        config::CrateType::Cdylib |
        config::CrateType::Dylib |
        config::CrateType::ProcMacro => {
            if !sess.target.target.options.dynamic_linking {
                return true
            }
            if sess.crt_static() && !sess.target.target.options.crt_static_allows_dylibs {
                return true
            }
        }
        _ => {}
    }
    if sess.target.target.options.only_cdylib {
        match crate_type {
            config::CrateType::ProcMacro | config::CrateType::Dylib => return true,
            _ => {}
        }
    }
    if !sess.target.target.options.executables {
        if crate_type == config::CrateType::Executable {
            return true
        }
    }

    false
}

pub fn linker_and_flavor(sess: &Session) -> (PathBuf, LinkerFlavor) {
    fn infer_from(
        sess: &Session,
        linker: Option<PathBuf>,
        flavor: Option<LinkerFlavor>,
    ) -> Option<(PathBuf, LinkerFlavor)> {
        match (linker, flavor) {
            (Some(linker), Some(flavor)) => Some((linker, flavor)),
            // only the linker flavor is known; use the default linker for the selected flavor
            (None, Some(flavor)) => Some((PathBuf::from(match flavor {
                LinkerFlavor::Em  => if cfg!(windows) { "emcc.bat" } else { "emcc" },
                LinkerFlavor::Gcc => "cc",
                LinkerFlavor::Ld => "ld",
                LinkerFlavor::Msvc => "link.exe",
                LinkerFlavor::Lld(_) => "lld",
            }), flavor)),
            (Some(linker), None) => {
                let stem = linker.file_stem().and_then(|stem| stem.to_str()).unwrap_or_else(|| {
                    sess.fatal("couldn't extract file stem from specified linker");
                }).to_owned();

                let flavor = if stem == "emcc" {
                    LinkerFlavor::Em
                } else if stem == "gcc" || stem.ends_with("-gcc") {
                    LinkerFlavor::Gcc
                } else if stem == "ld" || stem == "ld.lld" || stem.ends_with("-ld") {
                    LinkerFlavor::Ld
                } else if stem == "link" || stem == "lld-link" {
                    LinkerFlavor::Msvc
                } else if stem == "lld" || stem == "rust-lld" {
                    LinkerFlavor::Lld(sess.target.target.options.lld_flavor)
                } else {
                    // fall back to the value in the target spec
                    sess.target.target.linker_flavor
                };

                Some((linker, flavor))
            },
            (None, None) => None,
        }
    }

    // linker and linker flavor specified via command line have precedence over what the target
    // specification specifies
    if let Some(ret) = infer_from(
        sess,
        sess.opts.cg.linker.clone(),
        sess.opts.debugging_opts.linker_flavor,
    ) {
        return ret;
    }

    if let Some(ret) = infer_from(
        sess,
        sess.target.target.options.linker.clone().map(PathBuf::from),
        Some(sess.target.target.linker_flavor),
    ) {
        return ret;
    }

    bug!("Not enough information provided to determine how to invoke the linker");
}

// The third parameter is for env vars, used on windows to set up the
// path for MSVC to find its DLLs, and gcc to find its bundled
// toolchain
pub fn get_linker(sess: &Session, linker: &Path, flavor: LinkerFlavor) -> (PathBuf, Command) {
    let msvc_tool = windows_registry::find_tool(&sess.opts.target_triple.triple(), "link.exe");

    // If our linker looks like a batch script on Windows then to execute this
    // we'll need to spawn `cmd` explicitly. This is primarily done to handle
    // emscripten where the linker is `emcc.bat` and needs to be spawned as
    // `cmd /c emcc.bat ...`.
    //
    // This worked historically but is needed manually since #42436 (regression
    // was tagged as #42791) and some more info can be found on #44443 for
    // emscripten itself.
    let mut cmd = match linker.to_str() {
        Some(linker) if cfg!(windows) && linker.ends_with(".bat") => Command::bat_script(linker),
        _ => match flavor {
            LinkerFlavor::Lld(f) => Command::lld(linker, f),
            LinkerFlavor::Msvc
                if sess.opts.cg.linker.is_none() && sess.target.target.options.linker.is_none() =>
            {
                Command::new(msvc_tool.as_ref().map(|t| t.path()).unwrap_or(linker))
            },
            _ => Command::new(linker),
        }
    };

    // The compiler's sysroot often has some bundled tools, so add it to the
    // PATH for the child.
    let mut new_path = sess.host_filesearch(PathKind::All)
                           .get_tools_search_paths();
    let mut msvc_changed_path = false;
    if sess.target.target.options.is_like_msvc {
        if let Some(ref tool) = msvc_tool {
            cmd.args(tool.args());
            for &(ref k, ref v) in tool.env() {
                if k == "PATH" {
                    new_path.extend(env::split_paths(v));
                    msvc_changed_path = true;
                } else {
                    cmd.env(k, v);
                }
            }
        }
    }

    if !msvc_changed_path {
        if let Some(path) = env::var_os("PATH") {
            new_path.extend(env::split_paths(&path));
        }
    }
    cmd.env("PATH", env::join_paths(new_path).unwrap());

    (linker.to_path_buf(), cmd)
}


pub fn exec_linker(sess: &Session, cmd: &mut Command, out_filename: &Path, tmpdir: &Path)
    -> io::Result<Output>
{
    // When attempting to spawn the linker we run a risk of blowing out the
    // size limits for spawning a new process with respect to the arguments
    // we pass on the command line.
    //
    // Here we attempt to handle errors from the OS saying "your list of
    // arguments is too big" by reinvoking the linker again with an `@`-file
    // that contains all the arguments. The theory is that this is then
    // accepted on all linkers and the linker will read all its options out of
    // there instead of looking at the command line.
    if !cmd.very_likely_to_exceed_some_spawn_limit() {
        match cmd.command().stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(child) => {
                let output = child.wait_with_output();
                flush_linked_file(&output, out_filename)?;
                return output;
            }
            Err(ref e) if command_line_too_big(e) => {
                info!("command line to linker was too big: {}", e);
            }
            Err(e) => return Err(e)
        }
    }

    info!("falling back to passing arguments to linker via an @-file");
    let mut cmd2 = cmd.clone();
    let mut args = String::new();
    for arg in cmd2.take_args() {
        args.push_str(&Escape {
            arg: arg.to_str().unwrap(),
            is_like_msvc: sess.target.target.options.is_like_msvc,
        }.to_string());
        args.push_str("\n");
    }
    let file = tmpdir.join("linker-arguments");
    let bytes = if sess.target.target.options.is_like_msvc {
        let mut out = Vec::with_capacity((1 + args.len()) * 2);
        // start the stream with a UTF-16 BOM
        for c in iter::once(0xFEFF).chain(args.encode_utf16()) {
            // encode in little endian
            out.push(c as u8);
            out.push((c >> 8) as u8);
        }
        out
    } else {
        args.into_bytes()
    };
    fs::write(&file, &bytes)?;
    cmd2.arg(format!("@{}", file.display()));
    info!("invoking linker {:?}", cmd2);
    let output = cmd2.output();
    flush_linked_file(&output, out_filename)?;
    return output;

    #[cfg(unix)]
    fn flush_linked_file(_: &io::Result<Output>, _: &Path) -> io::Result<()> {
        Ok(())
    }

    #[cfg(windows)]
    fn flush_linked_file(command_output: &io::Result<Output>, out_filename: &Path)
        -> io::Result<()>
    {
        // On Windows, under high I/O load, output buffers are sometimes not flushed,
        // even long after process exit, causing nasty, non-reproducible output bugs.
        //
        // File::sync_all() calls FlushFileBuffers() down the line, which solves the problem.
        //
        // А full writeup of the original Chrome bug can be found at
        // randomascii.wordpress.com/2018/02/25/compiler-bug-linker-bug-windows-kernel-bug/amp

        if let &Ok(ref out) = command_output {
            if out.status.success() {
                if let Ok(of) = fs::OpenOptions::new().write(true).open(out_filename) {
                    of.sync_all()?;
                }
            }
        }

        Ok(())
    }

    #[cfg(unix)]
    fn command_line_too_big(err: &io::Error) -> bool {
        err.raw_os_error() == Some(::libc::E2BIG)
    }

    #[cfg(windows)]
    fn command_line_too_big(err: &io::Error) -> bool {
        const ERROR_FILENAME_EXCED_RANGE: i32 = 206;
        err.raw_os_error() == Some(ERROR_FILENAME_EXCED_RANGE)
    }

    struct Escape<'a> {
        arg: &'a str,
        is_like_msvc: bool,
    }

    impl<'a> fmt::Display for Escape<'a> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            if self.is_like_msvc {
                // This is "documented" at
                // https://msdn.microsoft.com/en-us/library/4xdcbak7.aspx
                //
                // Unfortunately there's not a great specification of the
                // syntax I could find online (at least) but some local
                // testing showed that this seemed sufficient-ish to catch
                // at least a few edge cases.
                write!(f, "\"")?;
                for c in self.arg.chars() {
                    match c {
                        '"' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
                write!(f, "\"")?;
            } else {
                // This is documented at https://linux.die.net/man/1/ld, namely:
                //
                // > Options in file are separated by whitespace. A whitespace
                // > character may be included in an option by surrounding the
                // > entire option in either single or double quotes. Any
                // > character (including a backslash) may be included by
                // > prefixing the character to be included with a backslash.
                //
                // We put an argument on each line, so all we need to do is
                // ensure the line is interpreted as one whole argument.
                for c in self.arg.chars() {
                    match c {
                        '\\' |
                        ' ' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
            }
            Ok(())
        }
    }
}
