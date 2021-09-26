use crate::base::ModuleData;
use rustc_ast::ptr::P;
use rustc_ast::{token, Attribute, Inline, Item};
use rustc_errors::{struct_span_err, DiagnosticBuilder};
use rustc_parse::new_parser_from_file;
use rustc_parse::validate_attr;
use rustc_session::parse::ParseSess;
use rustc_session::Session;
use rustc_span::symbol::{sym, Ident};
use rustc_span::Span;

use std::path::{self, Path, PathBuf};

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

crate struct ParsedExternalMod {
    pub items: Vec<P<Item>>,
    pub inner_span: Span,
    pub file_path: PathBuf,
    pub dir_path: PathBuf,
    pub dir_ownership: DirOwnership,
}

pub enum ModError<'a> {
    CircularInclusion(Vec<PathBuf>),
    ModInBlock(Option<Ident>),
    FileNotFound(Ident, PathBuf, PathBuf),
    MultipleCandidates(Ident, PathBuf, PathBuf),
    ParserError(DiagnosticBuilder<'a>),
}

crate fn parse_external_mod(
    sess: &Session,
    ident: Ident,
    span: Span, // The span to blame on errors.
    module: &ModuleData,
    mut dir_ownership: DirOwnership,
    attrs: &mut Vec<Attribute>,
) -> ParsedExternalMod {
    // We bail on the first error, but that error does not cause a fatal error... (1)
    let result: Result<_, ModError<'_>> = try {
        // Extract the file path and the new ownership.
        let mp = mod_file_path(sess, ident, &attrs, &module.dir_path, dir_ownership)?;
        dir_ownership = mp.dir_ownership;

        // Ensure file paths are acyclic.
        if let Some(pos) = module.file_path_stack.iter().position(|p| p == &mp.file_path) {
            Err(ModError::CircularInclusion(module.file_path_stack[pos..].to_vec()))?;
        }

        // Actually parse the external file as a module.
        let mut parser = new_parser_from_file(&sess.parse_sess, &mp.file_path, Some(span));
        let (mut inner_attrs, items, inner_span) =
            parser.parse_mod(&token::Eof).map_err(|err| ModError::ParserError(err))?;
        attrs.append(&mut inner_attrs);
        (items, inner_span, mp.file_path)
    };
    // (1) ...instead, we return a dummy module.
    let (items, inner_span, file_path) =
        result.map_err(|err| err.report(sess, span)).unwrap_or_default();

    // Extract the directory path for submodules of the module.
    let dir_path = file_path.parent().unwrap_or(&file_path).to_owned();

    ParsedExternalMod { items, inner_span, file_path, dir_path, dir_ownership }
}

crate fn mod_dir_path(
    sess: &Session,
    ident: Ident,
    attrs: &[Attribute],
    module: &ModuleData,
    mut dir_ownership: DirOwnership,
    inline: Inline,
) -> (PathBuf, DirOwnership) {
    match inline {
        Inline::Yes if let Some(file_path) = mod_file_path_from_attr(sess, attrs, &module.dir_path) => {
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
                    dir_path.push(&*ident.as_str());
                }
            }
            dir_path.push(&*ident.as_str());

            (dir_path, dir_ownership)
        }
        Inline::No => {
            // FIXME: This is a subset of `parse_external_mod` without actual parsing,
            // check whether the logic for unloaded, loaded and inline modules can be unified.
            let file_path = mod_file_path(sess, ident, &attrs, &module.dir_path, dir_ownership)
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
    let result = default_submod_path(&sess.parse_sess, ident, relative, dir_path);
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
fn mod_file_path_from_attr(
    sess: &Session,
    attrs: &[Attribute],
    dir_path: &Path,
) -> Option<PathBuf> {
    // Extract path string from first `#[path = "path_string"]` attribute.
    let first_path = attrs.iter().find(|at| at.has_name(sym::path))?;
    let path_string = match first_path.value_str() {
        Some(s) => s.as_str(),
        None => {
            // This check is here mainly to catch attempting to use a macro,
            // such as #[path = concat!(...)]. This isn't currently supported
            // because otherwise the InvocationCollector would need to defer
            // loading a module until the #[path] attribute was expanded, and
            // it doesn't support that (and would likely add a bit of
            // complexity). Usually bad forms are checked in AstValidator (via
            // `check_builtin_attribute`), but by the time that runs the macro
            // is expanded, and it doesn't give an error.
            validate_attr::emit_fatal_malformed_builtin_attribute(
                &sess.parse_sess,
                first_path,
                sym::path,
            );
        }
    };

    // On windows, the base path might have the form
    // `\\?\foo\bar` in which case it does not tolerate
    // mixed `/` and `\` separators, so canonicalize
    // `/` to `\`.
    #[cfg(windows)]
    let path_string = path_string.replace("/", "\\");

    Some(dir_path.join(&*path_string))
}

/// Returns a path to a module.
// Public for rustfmt usage.
pub fn default_submod_path<'a>(
    sess: &'a ParseSess,
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

    let mod_name = ident.name.to_string();
    let default_path_str = format!("{}{}.rs", relative_prefix, mod_name);
    let secondary_path_str =
        format!("{}{}{}mod.rs", relative_prefix, mod_name, path::MAIN_SEPARATOR);
    let default_path = dir_path.join(&default_path_str);
    let secondary_path = dir_path.join(&secondary_path_str);
    let default_exists = sess.source_map().file_exists(&default_path);
    let secondary_exists = sess.source_map().file_exists(&secondary_path);

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
    fn report(self, sess: &Session, span: Span) {
        let diag = &sess.parse_sess.span_diagnostic;
        match self {
            ModError::CircularInclusion(file_paths) => {
                let mut msg = String::from("circular modules: ");
                for file_path in &file_paths {
                    msg.push_str(&file_path.display().to_string());
                    msg.push_str(" -> ");
                }
                msg.push_str(&file_paths[0].display().to_string());
                diag.struct_span_err(span, &msg)
            }
            ModError::ModInBlock(ident) => {
                let msg = "cannot declare a non-inline module inside a block unless it has a path attribute";
                let mut err = diag.struct_span_err(span, msg);
                if let Some(ident) = ident {
                    let note =
                        format!("maybe `use` the module `{}` instead of redeclaring it", ident);
                    err.span_note(span, &note);
                }
                err
            }
            ModError::FileNotFound(ident, default_path, secondary_path) => {
                let mut err = struct_span_err!(
                    diag,
                    span,
                    E0583,
                    "file not found for module `{}`",
                    ident,
                );
                err.help(&format!(
                    "to create the module `{}`, create file \"{}\" or \"{}\"",
                    ident,
                    default_path.display(),
                    secondary_path.display(),
                ));
                err
            }
            ModError::MultipleCandidates(ident, default_path, secondary_path) => {
                let mut err = struct_span_err!(
                    diag,
                    span,
                    E0761,
                    "file for module `{}` found at both \"{}\" and \"{}\"",
                    ident,
                    default_path.display(),
                    secondary_path.display(),
                );
                err.help("delete or rename one of them to remove the ambiguity");
                err
            }
            ModError::ParserError(err) => err,
        }.emit()
    }
}
