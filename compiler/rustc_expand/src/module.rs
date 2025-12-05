use std::iter::once;
use std::path::{self, Path, PathBuf};

use rustc_ast::{AttrVec, Attribute, Inline, Item, ModSpans};
use rustc_attr_parsing::validate_attr;
use rustc_errors::{Diag, ErrorGuaranteed};
use rustc_parse::lexer::StripTokens;
use rustc_parse::{exp, new_parser_from_file, unwrap_or_emit_fatal};
use rustc_session::Session;
use rustc_session::parse::ParseSess;
use rustc_span::{Ident, Span, sym};
use thin_vec::ThinVec;

use crate::base::ModuleData;
use crate::errors::{
    ModuleCircular, ModuleFileNotFound, ModuleInBlock, ModuleInBlockName, ModuleMultipleCandidates,
};

#[derive(Copy, Clone)]
pub enum DirOwnership {
    Owned {
        // None if `mod.rs`, `Some("foo")` if we're in `foo.rs`.
        relative: Option<Ident>,
    },
    UnownedViaBlock,
}

// Public for rustfmt usage.
pub struct ModulePathSuccess {
    pub file_path: PathBuf,
    pub dir_ownership: DirOwnership,
}

pub(crate) struct ParsedExternalMod {
    pub items: ThinVec<Box<Item>>,
    pub spans: ModSpans,
    pub file_path: PathBuf,
    pub dir_path: PathBuf,
    pub dir_ownership: DirOwnership,
    pub had_parse_error: Result<(), ErrorGuaranteed>,
}

pub enum ModError<'a> {
    CircularInclusion(Vec<PathBuf>),
    ModInBlock(Option<Ident>),
    FileNotFound(Ident, PathBuf, PathBuf),
    MultipleCandidates(Ident, PathBuf, PathBuf),
    ParserError(Diag<'a>),
}

pub(crate) fn parse_external_mod(
    sess: &Session,
    ident: Ident,
    span: Span, // The span to blame on errors.
    module: &ModuleData,
    mut dir_ownership: DirOwnership,
    attrs: &mut AttrVec,
) -> ParsedExternalMod {
    // We bail on the first error, but that error does not cause a fatal error... (1)
    let result: Result<_, ModError<'_>> = try {
        // Extract the file path and the new ownership.
        let mp = mod_file_path(sess, ident, attrs, &module.dir_path, dir_ownership)?;
        dir_ownership = mp.dir_ownership;

        // Ensure file paths are acyclic.
        if let Some(pos) = module.file_path_stack.iter().position(|p| p == &mp.file_path) {
            do yeet ModError::CircularInclusion(module.file_path_stack[pos..].to_vec());
        }

        // Actually parse the external file as a module.
        let mut parser = unwrap_or_emit_fatal(new_parser_from_file(
            &sess.psess,
            &mp.file_path,
            StripTokens::ShebangAndFrontmatter,
            Some(span),
        ));
        let (inner_attrs, items, inner_span) =
            parser.parse_mod(exp!(Eof)).map_err(|err| ModError::ParserError(err))?;
        attrs.extend(inner_attrs);
        (items, inner_span, mp.file_path)
    };

    // (1) ...instead, we return a dummy module.
    let ((items, spans, file_path), had_parse_error) = match result {
        Err(err) => (Default::default(), Err(err.report(sess, span))),
        Ok(result) => (result, Ok(())),
    };

    // Extract the directory path for submodules of the module.
    let dir_path = file_path.parent().unwrap_or(&file_path).to_owned();

    ParsedExternalMod { items, spans, file_path, dir_path, dir_ownership, had_parse_error }
}

pub(crate) fn mod_dir_path(
    sess: &Session,
    ident: Ident,
    attrs: &[Attribute],
    module: &ModuleData,
    mut dir_ownership: DirOwnership,
    inline: Inline,
) -> (PathBuf, DirOwnership) {
    match inline {
        Inline::Yes
            if let Some(file_path) = mod_file_path_from_attr(sess, attrs, &module.dir_path) =>
        {
            // For inline modules file path from `#[path]` is actually the directory path
            // for historical reasons, so we don't pop the last segment here.
            (file_path, DirOwnership::Owned { relative: None })
        }
        Inline::Yes => {
            // We have to push on the current module name in the case of relative
            // paths in order to ensure that any additional module paths from inline
            // `mod x { ... }` come after the relative extension.
            //
            // For example, a `mod z { ... }` inside `x/y.rs` should set the current
            // directory path to `/x/y/z`, not `/x/z` with a relative offset of `y`.
            let mut dir_path = module.dir_path.clone();
            if let DirOwnership::Owned { relative } = &mut dir_ownership {
                if let Some(ident) = relative.take() {
                    // Remove the relative offset.
                    dir_path.push(ident.as_str());
                }
            }
            dir_path.push(ident.as_str());

            (dir_path, dir_ownership)
        }
        Inline::No { .. } => {
            // FIXME: This is a subset of `parse_external_mod` without actual parsing,
            // check whether the logic for unloaded, loaded and inline modules can be unified.
            let file_path = mod_file_path(sess, ident, attrs, &module.dir_path, dir_ownership)
                .map(|mp| {
                    dir_ownership = mp.dir_ownership;
                    mp.file_path
                })
                .unwrap_or_default();

            // Extract the directory path for submodules of the module.
            let dir_path = file_path.parent().unwrap_or(&file_path).to_owned();

            (dir_path, dir_ownership)
        }
    }
}

fn mod_file_path<'a>(
    sess: &'a Session,
    ident: Ident,
    attrs: &[Attribute],
    dir_path: &Path,
    dir_ownership: DirOwnership,
) -> Result<ModulePathSuccess, ModError<'a>> {
    if let Some(file_path) = mod_file_path_from_attr(sess, attrs, dir_path) {
        // All `#[path]` files are treated as though they are a `mod.rs` file.
        // This means that `mod foo;` declarations inside `#[path]`-included
        // files are siblings,
        //
        // Note that this will produce weirdness when a file named `foo.rs` is
        // `#[path]` included and contains a `mod foo;` declaration.
        // If you encounter this, it's your own darn fault :P
        let dir_ownership = DirOwnership::Owned { relative: None };
        return Ok(ModulePathSuccess { file_path, dir_ownership });
    }

    let relative = match dir_ownership {
        DirOwnership::Owned { relative } => relative,
        DirOwnership::UnownedViaBlock => None,
    };
    let result = default_submod_path(&sess.psess, ident, relative, dir_path);
    match dir_ownership {
        DirOwnership::Owned { .. } => result,
        DirOwnership::UnownedViaBlock => Err(ModError::ModInBlock(match result {
            Ok(_) | Err(ModError::MultipleCandidates(..)) => Some(ident),
            _ => None,
        })),
    }
}

/// Derive a submodule path from the first found `#[path = "path_string"]`.
/// The provided `dir_path` is joined with the `path_string`.
pub(crate) fn mod_file_path_from_attr(
    sess: &Session,
    attrs: &[Attribute],
    dir_path: &Path,
) -> Option<PathBuf> {
    // Extract path string from first `#[path = "path_string"]` attribute.
    let first_path = attrs.iter().find(|at| at.has_name(sym::path))?;
    let Some(path_sym) = first_path.value_str() else {
        // This check is here mainly to catch attempting to use a macro,
        // such as `#[path = concat!(...)]`. This isn't supported because
        // otherwise the `InvocationCollector` would need to defer loading
        // a module until the `#[path]` attribute was expanded, and it
        // doesn't support that (and would likely add a bit of complexity).
        // Usually bad forms are checked during semantic analysis via
        // `TyCtxt::check_mod_attrs`), but by the time that runs the macro
        // is expanded, and it doesn't give an error.
        validate_attr::emit_fatal_malformed_builtin_attribute(&sess.psess, first_path, sym::path);
    };

    let path_str = path_sym.as_str();

    // On windows, the base path might have the form
    // `\\?\foo\bar` in which case it does not tolerate
    // mixed `/` and `\` separators, so canonicalize
    // `/` to `\`.
    #[cfg(windows)]
    let path_str = path_str.replace("/", "\\");

    Some(dir_path.join(path_str))
}

/// Returns a path to a module.
// Public for rustfmt usage.
pub fn default_submod_path<'a>(
    psess: &'a ParseSess,
    ident: Ident,
    relative: Option<Ident>,
    dir_path: &Path,
) -> Result<ModulePathSuccess, ModError<'a>> {
    // If we're in a foo.rs file instead of a mod.rs file,
    // we need to look for submodules in
    // `./foo/<ident>.rs` and `./foo/<ident>/mod.rs` rather than
    // `./<ident>.rs` and `./<ident>/mod.rs`.
    let relative_prefix_string;
    let relative_prefix = if let Some(ident) = relative {
        relative_prefix_string = format!("{}{}", ident.name, path::MAIN_SEPARATOR);
        &relative_prefix_string
    } else {
        ""
    };

    let default_path_str = format!("{}{}.rs", relative_prefix, ident.name);
    let secondary_path_str =
        format!("{}{}{}mod.rs", relative_prefix, ident.name, path::MAIN_SEPARATOR);
    let default_path = dir_path.join(&default_path_str);
    let secondary_path = dir_path.join(&secondary_path_str);
    let default_exists = psess.source_map().file_exists(&default_path);
    let secondary_exists = psess.source_map().file_exists(&secondary_path);

    match (default_exists, secondary_exists) {
        (true, false) => Ok(ModulePathSuccess {
            file_path: default_path,
            dir_ownership: DirOwnership::Owned { relative: Some(ident) },
        }),
        (false, true) => Ok(ModulePathSuccess {
            file_path: secondary_path,
            dir_ownership: DirOwnership::Owned { relative: None },
        }),
        (false, false) => Err(ModError::FileNotFound(ident, default_path, secondary_path)),
        (true, true) => Err(ModError::MultipleCandidates(ident, default_path, secondary_path)),
    }
}

impl ModError<'_> {
    fn report(self, sess: &Session, span: Span) -> ErrorGuaranteed {
        match self {
            ModError::CircularInclusion(file_paths) => {
                let path_to_string = |path: &PathBuf| path.display().to_string();

                let paths = file_paths
                    .iter()
                    .map(path_to_string)
                    .chain(once(path_to_string(&file_paths[0])))
                    .collect::<Vec<_>>();

                let modules = paths.join(" -> ");

                sess.dcx().emit_err(ModuleCircular { span, modules })
            }
            ModError::ModInBlock(ident) => sess.dcx().emit_err(ModuleInBlock {
                span,
                name: ident.map(|name| ModuleInBlockName { span, name }),
            }),
            ModError::FileNotFound(name, default_path, secondary_path) => {
                sess.dcx().emit_err(ModuleFileNotFound {
                    span,
                    name,
                    default_path: default_path.display().to_string(),
                    secondary_path: secondary_path.display().to_string(),
                })
            }
            ModError::MultipleCandidates(name, default_path, secondary_path) => {
                sess.dcx().emit_err(ModuleMultipleCandidates {
                    span,
                    name,
                    default_path: default_path.display().to_string(),
                    secondary_path: secondary_path.display().to_string(),
                })
            }
            ModError::ParserError(err) => err.emit(),
        }
    }
}
