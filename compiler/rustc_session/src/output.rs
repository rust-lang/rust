//! Related to out filenames of compilation (e.g. binaries).

use std::path::Path;

use rustc_span::{Span, Symbol};

use crate::Session;
use crate::config::{CrateType, OutFileName, OutputFilenames, OutputType};
use crate::errors::{CrateNameEmpty, FileIsNotWriteable, InvalidCharacterInCrateName};

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

/// Validate the given crate name.
///
/// Note that this validation is more permissive than identifier parsing. It considers
/// non-empty sequences of alphanumeric and underscore characters to be valid crate names.
/// Most notably, it accepts names starting with a numeric character like `0`!
///
/// Furthermore, this shouldn't be taken as the canonical crate name validator.
/// Other places may use a more restrictive grammar (e.g., identifier or ASCII identifier).
pub fn validate_crate_name(sess: &Session, crate_name: Symbol, span: Option<Span>) {
    let mut guar = None;

    if crate_name.is_empty() {
        guar = Some(sess.dcx().emit_err(CrateNameEmpty { span }));
    }

    for c in crate_name.as_str().chars() {
        if c.is_alphanumeric() || c == '_' {
            continue;
        }
        guar = Some(sess.dcx().emit_err(InvalidCharacterInCrateName {
            span,
            character: c,
            crate_name,
        }));
    }

    if let Some(guar) = guar {
        guar.raise_fatal();
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
        CrateType::Cdylib | CrateType::ProcMacro | CrateType::Dylib | CrateType::Sdylib => {
            let (prefix, suffix) = (&sess.target.dll_prefix, &sess.target.dll_suffix);
            OutFileName::Real(outputs.out_directory.join(&format!("{prefix}{libname}{suffix}")))
        }
        CrateType::StaticLib => {
            let (prefix, suffix) = sess.staticlib_components(false);
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
