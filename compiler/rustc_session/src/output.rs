//! Related to out filenames of compilation (e.g. binaries).

use std::path::Path;

use rustc_ast::{self as ast, attr};
use rustc_errors::FatalError;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};

use crate::Session;
use crate::config::{self, CrateType, Input, OutFileName, OutputFilenames, OutputType};
use crate::errors::{
    self, CrateNameDoesNotMatch, CrateNameEmpty, CrateNameInvalid, FileIsNotWriteable,
    InvalidCharacterInCrateName, InvalidCrateNameHelp,
};

pub fn out_filename(
    sess: &Session,
    crate_type: CrateType,
    outputs: &OutputFilenames,
    crate_name: Symbol,
) -> OutFileName {
    let default_filename = filename_for_input(sess, crate_type, crate_name, outputs);
    let out_filename = outputs
        .outputs
        .get(&OutputType::Exe)
        .and_then(|s| s.to_owned())
        .or_else(|| outputs.single_output_file.clone())
        .unwrap_or(default_filename);

    if let OutFileName::Real(ref path) = out_filename {
        check_file_is_writeable(path, sess);
    }

    out_filename
}

/// Make sure files are writeable. Mac, FreeBSD, and Windows system linkers
/// check this already -- however, the Linux linker will happily overwrite a
/// read-only file. We should be consistent.
pub fn check_file_is_writeable(file: &Path, sess: &Session) {
    if !is_writeable(file) {
        sess.dcx().emit_fatal(FileIsNotWriteable { file });
    }
}

fn is_writeable(p: &Path) -> bool {
    match p.metadata() {
        Err(..) => true,
        Ok(m) => !m.permissions().readonly(),
    }
}

pub fn find_crate_name(sess: &Session, attrs: &[ast::Attribute]) -> Symbol {
    let validate = |s: Symbol, span: Option<Span>| {
        validate_crate_name(sess, s, span);
        s
    };

    // Look in attributes 100% of the time to make sure the attribute is marked
    // as used. After doing this, however, we still prioritize a crate name from
    // the command line over one found in the #[crate_name] attribute. If we
    // find both we ensure that they're the same later on as well.
    let attr_crate_name =
        attr::find_by_name(attrs, sym::crate_name).and_then(|at| at.value_str().map(|s| (at, s)));

    if let Some(ref s) = sess.opts.crate_name {
        let s = Symbol::intern(s);
        if let Some((attr, name)) = attr_crate_name {
            if name != s {
                sess.dcx().emit_err(CrateNameDoesNotMatch { span: attr.span, s, name });
            }
        }
        return validate(s, None);
    }

    if let Some((attr, s)) = attr_crate_name {
        return validate(s, Some(attr.span));
    }
    if let Input::File(ref path) = sess.io.input {
        if let Some(s) = path.file_stem().and_then(|s| s.to_str()) {
            if s.starts_with('-') {
                sess.dcx().emit_err(CrateNameInvalid { s });
            } else {
                return validate(Symbol::intern(&s.replace('-', "_")), None);
            }
        }
    }

    Symbol::intern("rust_out")
}

pub fn validate_crate_name(sess: &Session, s: Symbol, sp: Option<Span>) {
    let mut err_count = 0;
    {
        if s.is_empty() {
            err_count += 1;
            sess.dcx().emit_err(CrateNameEmpty { span: sp });
        }
        for c in s.as_str().chars() {
            if c.is_alphanumeric() {
                continue;
            }
            if c == '_' {
                continue;
            }
            err_count += 1;
            sess.dcx().emit_err(InvalidCharacterInCrateName {
                span: sp,
                character: c,
                crate_name: s,
                crate_name_help: if sp.is_none() {
                    Some(InvalidCrateNameHelp::AddCrateName)
                } else {
                    None
                },
            });
        }
    }

    if err_count > 0 {
        FatalError.raise();
    }
}

pub fn filename_for_metadata(sess: &Session, outputs: &OutputFilenames) -> OutFileName {
    let out_filename = outputs.path(OutputType::Metadata);
    if let OutFileName::Real(ref path) = out_filename {
        check_file_is_writeable(path, sess);
    }
    out_filename
}

pub fn filename_for_input(
    sess: &Session,
    crate_type: CrateType,
    crate_name: Symbol,
    outputs: &OutputFilenames,
) -> OutFileName {
    let libname = format!("{}{}", crate_name, sess.opts.cg.extra_filename);

    match crate_type {
        CrateType::Rlib => {
            OutFileName::Real(outputs.out_directory.join(&format!("lib{libname}.rlib")))
        }
        CrateType::Cdylib | CrateType::ProcMacro | CrateType::Dylib => {
            let (prefix, suffix) = (&sess.target.dll_prefix, &sess.target.dll_suffix);
            OutFileName::Real(outputs.out_directory.join(&format!("{prefix}{libname}{suffix}")))
        }
        CrateType::Staticlib => {
            let (prefix, suffix) = (&sess.target.staticlib_prefix, &sess.target.staticlib_suffix);
            OutFileName::Real(outputs.out_directory.join(&format!("{prefix}{libname}{suffix}")))
        }
        CrateType::Executable => {
            let suffix = &sess.target.exe_suffix;
            let out_filename = outputs.path(OutputType::Exe);
            if let OutFileName::Real(ref path) = out_filename {
                if suffix.is_empty() {
                    out_filename
                } else {
                    OutFileName::Real(path.with_extension(&suffix[1..]))
                }
            } else {
                out_filename
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
pub fn default_output_for_target(sess: &Session) -> CrateType {
    if !sess.target.executables { CrateType::Staticlib } else { CrateType::Executable }
}

/// Checks if target supports crate_type as output
pub fn invalid_output_for_target(sess: &Session, crate_type: CrateType) -> bool {
    if let CrateType::Cdylib | CrateType::Dylib | CrateType::ProcMacro = crate_type {
        if !sess.target.dynamic_linking {
            return true;
        }
        if sess.crt_static(Some(crate_type)) && !sess.target.crt_static_allows_dylibs {
            return true;
        }
    }
    if let CrateType::ProcMacro | CrateType::Dylib = crate_type
        && sess.target.only_cdylib
    {
        return true;
    }
    if let CrateType::Executable = crate_type
        && !sess.target.executables
    {
        return true;
    }

    false
}

pub const CRATE_TYPES: &[(Symbol, CrateType)] = &[
    (sym::rlib, CrateType::Rlib),
    (sym::dylib, CrateType::Dylib),
    (sym::cdylib, CrateType::Cdylib),
    (sym::lib, config::default_lib_output()),
    (sym::staticlib, CrateType::Staticlib),
    (sym::proc_dash_macro, CrateType::ProcMacro),
    (sym::bin, CrateType::Executable),
];

pub fn categorize_crate_type(s: Symbol) -> Option<CrateType> {
    Some(CRATE_TYPES.iter().find(|(key, _)| *key == s)?.1)
}

pub fn collect_crate_types(session: &Session, attrs: &[ast::Attribute]) -> Vec<CrateType> {
    // If we're generating a test executable, then ignore all other output
    // styles at all other locations
    if session.opts.test {
        return vec![CrateType::Executable];
    }

    // Only check command line flags if present. If no types are specified by
    // command line, then reuse the empty `base` Vec to hold the types that
    // will be found in crate attributes.
    // JUSTIFICATION: before wrapper fn is available
    #[allow(rustc::bad_opt_access)]
    let mut base = session.opts.crate_types.clone();
    if base.is_empty() {
        let attr_types = attrs.iter().filter_map(|a| {
            if a.has_name(sym::crate_type)
                && let Some(s) = a.value_str()
            {
                categorize_crate_type(s)
            } else {
                None
            }
        });
        base.extend(attr_types);
        if base.is_empty() {
            base.push(default_output_for_target(session));
        } else {
            base.sort();
            base.dedup();
        }
    }

    base.retain(|crate_type| {
        if invalid_output_for_target(session, *crate_type) {
            session.dcx().emit_warn(errors::UnsupportedCrateTypeForTarget {
                crate_type: *crate_type,
                target_triple: &session.opts.target_triple,
            });
            false
        } else {
            true
        }
    });

    base
}
