// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::session::config::{self, OutputFilenames, Input, OutputType};
use rustc::session::Session;
use rustc::middle::cstore::{self, LinkMeta};
use rustc::hir::svh::Svh;
use std::path::{Path, PathBuf};
use syntax::{ast, attr};
use syntax_pos::Span;

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

pub fn build_link_meta(crate_hash: Svh) -> LinkMeta {
    let r = LinkMeta {
        crate_hash,
    };
    info!("{:?}", r);
    return r;
}

pub fn find_crate_name(sess: Option<&Session>,
                       attrs: &[ast::Attribute],
                       input: &Input) -> String {
    let validate = |s: String, span: Option<Span>| {
        cstore::validate_crate_name(sess, &s, span);
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

pub fn filename_for_input(sess: &Session,
                          crate_type: config::CrateType,
                          crate_name: &str,
                          outputs: &OutputFilenames) -> PathBuf {
    let libname = format!("{}{}", crate_name, sess.opts.cg.extra_filename);

    match crate_type {
        config::CrateTypeRlib => {
            outputs.out_directory.join(&format!("lib{}.rlib", libname))
        }
        config::CrateTypeCdylib |
        config::CrateTypeProcMacro |
        config::CrateTypeDylib => {
            let (prefix, suffix) = (&sess.target.target.options.dll_prefix,
                                    &sess.target.target.options.dll_suffix);
            outputs.out_directory.join(&format!("{}{}{}", prefix, libname,
                                                suffix))
        }
        config::CrateTypeStaticlib => {
            let (prefix, suffix) = (&sess.target.target.options.staticlib_prefix,
                                    &sess.target.target.options.staticlib_suffix);
            outputs.out_directory.join(&format!("{}{}{}", prefix, libname,
                                                suffix))
        }
        config::CrateTypeExecutable => {
            let suffix = &sess.target.target.options.exe_suffix;
            let out_filename = outputs.path(OutputType::Exe);
            if suffix.is_empty() {
                out_filename.to_path_buf()
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
/// It is CrateTypeExecutable for all platforms but iOS as there is no
/// way to run iOS binaries anyway without jailbreaking and
/// interaction with Rust code through static library is the only
/// option for now
pub fn default_output_for_target(sess: &Session) -> config::CrateType {
    if !sess.target.target.options.executables {
        config::CrateTypeStaticlib
    } else {
        config::CrateTypeExecutable
    }
}

/// Checks if target supports crate_type as output
pub fn invalid_output_for_target(sess: &Session,
                                 crate_type: config::CrateType) -> bool {
    match crate_type {
        config::CrateTypeCdylib |
        config::CrateTypeDylib |
        config::CrateTypeProcMacro => {
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
            config::CrateTypeProcMacro | config::CrateTypeDylib => return true,
            _ => {}
        }
    }
    if !sess.target.target.options.executables {
        if crate_type == config::CrateTypeExecutable {
            return true
        }
    }

    false
}
