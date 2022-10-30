//! Related to out filenames of compilation (e.g. save analysis, binaries).
use crate::config::{CrateType, Input, OutputFilenames, OutputType};
use crate::errors::{
    CrateNameDoesNotMatch, CrateNameEmpty, CrateNameInvalid, FileIsNotWriteable,
    InvalidCharacterInCrateName,
};
use crate::Session;
use rustc_ast as ast;
use rustc_span::symbol::sym;
use rustc_span::Span;
use std::path::{Path, PathBuf};

pub fn out_filename(
    sess: &Session,
    crate_type: CrateType,
    outputs: &OutputFilenames,
    crate_name: &str,
) -> PathBuf {
    let default_filename = filename_for_input(sess, crate_type, crate_name, outputs);
    let out_filename = outputs
        .outputs
        .get(&OutputType::Exe)
        .and_then(|s| s.to_owned())
        .or_else(|| outputs.single_output_file.clone())
        .unwrap_or(default_filename);

    check_file_is_writeable(&out_filename, sess);

    out_filename
}

/// Make sure files are writeable.  Mac, FreeBSD, and Windows system linkers
/// check this already -- however, the Linux linker will happily overwrite a
/// read-only file.  We should be consistent.
pub fn check_file_is_writeable(file: &Path, sess: &Session) {
    if !is_writeable(file) {
        sess.emit_fatal(FileIsNotWriteable { file });
    }
}

fn is_writeable(p: &Path) -> bool {
    match p.metadata() {
        Err(..) => true,
        Ok(m) => !m.permissions().readonly(),
    }
}

pub fn find_crate_name(sess: &Session, attrs: &[ast::Attribute], input: &Input) -> String {
    let validate = |s: String, span: Option<Span>| {
        validate_crate_name(sess, &s, span);
        s
    };

    // Look in attributes 100% of the time to make sure the attribute is marked
    // as used. After doing this, however, we still prioritize a crate name from
    // the command line over one found in the #[crate_name] attribute. If we
    // find both we ensure that they're the same later on as well.
    let attr_crate_name =
        sess.find_by_name(attrs, sym::crate_name).and_then(|at| at.value_str().map(|s| (at, s)));

    if let Some(ref s) = sess.opts.crate_name {
        if let Some((attr, name)) = attr_crate_name {
            if name.as_str() != s {
                sess.emit_err(CrateNameDoesNotMatch { span: attr.span, s, name });
            }
        }
        return validate(s.clone(), None);
    }

    if let Some((attr, s)) = attr_crate_name {
        return validate(s.to_string(), Some(attr.span));
    }
    if let Input::File(ref path) = *input {
        if let Some(s) = path.file_stem().and_then(|s| s.to_str()) {
            if s.starts_with('-') {
                sess.emit_err(CrateNameInvalid { s });
            } else {
                return validate(s.replace('-', "_"), None);
            }
        }
    }

    "rust_out".to_string()
}

pub fn validate_crate_name(sess: &Session, s: &str, sp: Option<Span>) {
    let mut err_count = 0;
    {
        if s.is_empty() {
            err_count += 1;
            sess.emit_err(CrateNameEmpty { span: sp });
        }
        for c in s.chars() {
            if c.is_alphanumeric() {
                continue;
            }
            if c == '_' {
                continue;
            }
            err_count += 1;
            sess.emit_err(InvalidCharacterInCrateName { span: sp, character: c, crate_name: s });
        }
    }

    if err_count > 0 {
        sess.abort_if_errors();
    }
}

pub fn filename_for_metadata(
    sess: &Session,
    crate_name: &str,
    outputs: &OutputFilenames,
) -> PathBuf {
    // If the command-line specified the path, use that directly.
    if let Some(Some(out_filename)) = sess.opts.output_types.get(&OutputType::Metadata) {
        return out_filename.clone();
    }

    let libname = format!("{}{}", crate_name, sess.opts.cg.extra_filename);

    let out_filename = outputs
        .single_output_file
        .clone()
        .unwrap_or_else(|| outputs.out_directory.join(&format!("lib{libname}.rmeta")));

    check_file_is_writeable(&out_filename, sess);

    out_filename
}

pub fn filename_for_input(
    sess: &Session,
    crate_type: CrateType,
    crate_name: &str,
    outputs: &OutputFilenames,
) -> PathBuf {
    let libname = format!("{}{}", crate_name, sess.opts.cg.extra_filename);

    match crate_type {
        CrateType::Rlib => outputs.out_directory.join(&format!("lib{libname}.rlib")),
        CrateType::Cdylib | CrateType::ProcMacro | CrateType::Dylib => {
            let (prefix, suffix) = (&sess.target.dll_prefix, &sess.target.dll_suffix);
            outputs.out_directory.join(&format!("{prefix}{libname}{suffix}"))
        }
        CrateType::Staticlib => {
            let (prefix, suffix) = (&sess.target.staticlib_prefix, &sess.target.staticlib_suffix);
            outputs.out_directory.join(&format!("{prefix}{libname}{suffix}"))
        }
        CrateType::Executable => {
            let suffix = &sess.target.exe_suffix;
            let out_filename = outputs.path(OutputType::Exe);
            if suffix.is_empty() { out_filename } else { out_filename.with_extension(&suffix[1..]) }
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
    if let CrateType::ProcMacro | CrateType::Dylib = crate_type && sess.target.only_cdylib {
        return true;
    }
    if let CrateType::Executable = crate_type && !sess.target.executables {
        return true;
    }

    false
}
