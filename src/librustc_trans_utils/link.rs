// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::session::config::{self, /*NoDebugInfo,*/ OutputFilenames, Input, OutputType};
/*use rustc::session::filesearch;
use rustc::session::search_paths::PathKind;
*/use rustc::session::Session;
use rustc::middle::cstore;/*::{self, LinkMeta, NativeLibrary, LibSource, LinkagePreference,
                            NativeLibraryKind};*/
/*use rustc::middle::dependency_format::Linkage;
use rustc::util::common::time;
use rustc::util::fs::fix_windows_verbatim_for_gcc;
use rustc::dep_graph::{DepKind, DepNode};
use rustc::hir::def_id::CrateNum;
use rustc::hir::svh::Svh;
use rustc_back::tempdir::TempDir;
use rustc_back::{PanicStrategy, RelroLevel};
use rustc_incremental::IncrementalHashesMap;*/

/*use std::ascii;
use std::char;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::mem;
*/use std::path::PathBuf;/*{Path, PathBuf};
use std::process::Command;
use std::str;*/
use syntax::ast;
//use syntax::attr;
use syntax_pos::Span;

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
    let attr_crate_name = attrs.iter().find(|at| at.check_name("crate_name"))
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
    match (sess.target.target.options.dynamic_linking,
           sess.target.target.options.executables, crate_type) {
        (false, _, config::CrateTypeCdylib) |
        (false, _, config::CrateTypeProcMacro) |
        (false, _, config::CrateTypeDylib) => true,
        (_, false, config::CrateTypeExecutable) => true,
        _ => false
    }
}
