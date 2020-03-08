use super::item::ItemInfo;
use super::Parser;

use crate::{new_sub_parser_from_file, DirectoryOwnership};

use rustc_ast::ast::{self, Attribute, Crate, Ident, ItemKind, Mod};
use rustc_ast::attr;
use rustc_ast::token::{self, TokenKind};
use rustc_errors::{struct_span_err, PResult};
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{FileName, Span};
use rustc_span::symbol::sym;

use std::path::{self, Path, PathBuf};

/// Information about the path to a module.
// Public for rustfmt usage.
pub struct ModulePath<'a> {
    name: String,
    path_exists: bool,
    pub result: PResult<'a, ModulePathSuccess>,
}

// Public for rustfmt usage.
pub struct ModulePathSuccess {
    pub path: PathBuf,
    pub ownership: DirectoryOwnership,
}

impl<'a> Parser<'a> {
    /// Parses a source module as a crate. This is the main entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> PResult<'a, Crate> {
        let lo = self.token.span;
        let (module, attrs) = self.parse_mod(&token::Eof)?;
        let span = lo.to(self.token.span);
        let proc_macros = Vec::new(); // Filled in by `proc_macro_harness::inject()`.
        Ok(ast::Crate { attrs, module, span, proc_macros })
    }

    /// Parses a `mod <foo> { ... }` or `mod <foo>;` item.
    pub(super) fn parse_item_mod(&mut self, attrs: &mut Vec<Attribute>) -> PResult<'a, ItemInfo> {
        let in_cfg = crate::config::process_configure_mod(self.sess, self.cfg_mods, attrs);

        let id = self.parse_ident()?;
        let (module, mut inner_attrs) = if self.eat(&token::Semi) {
            if in_cfg && self.recurse_into_file_modules {
                // This mod is in an external file. Let's go get it!
                let dir = &self.directory;
                submod_path(self.sess, id, &attrs, dir.ownership, &dir.path)
                    .and_then(|r| eval_src_mod(self.sess, self.cfg_mods, r.path, r.ownership, id))
                    .map_err(|mut err| err.emit())
                    .unwrap_or_default()
            } else {
                Default::default()
            }
        } else {
            let old_directory = self.directory.clone();
            push_directory(id, &attrs, &mut self.directory.ownership, &mut self.directory.path);

            self.expect(&token::OpenDelim(token::Brace))?;
            let module = self.parse_mod(&token::CloseDelim(token::Brace))?;

            self.directory = old_directory;
            module
        };
        attrs.append(&mut inner_attrs);
        Ok((id, ItemKind::Mod(module)))
    }

    /// Parses the contents of a module (inner attributes followed by module items).
    fn parse_mod(&mut self, term: &TokenKind) -> PResult<'a, (Mod, Vec<Attribute>)> {
        let lo = self.token.span;
        let attrs = self.parse_inner_attributes()?;
        let module = self.parse_mod_items(term, lo)?;
        Ok((module, attrs))
    }

    /// Given a termination token, parses all of the items in a module.
    fn parse_mod_items(&mut self, term: &TokenKind, inner_lo: Span) -> PResult<'a, Mod> {
        let mut items = vec![];
        while let Some(item) = self.parse_item()? {
            items.push(item);
            self.maybe_consume_incorrect_semicolon(&items);
        }

        if !self.eat(term) {
            let token_str = super::token_descr(&self.token);
            if !self.maybe_consume_incorrect_semicolon(&items) {
                let msg = &format!("expected item, found {}", token_str);
                let mut err = self.struct_span_err(self.token.span, msg);
                err.span_label(self.token.span, "expected item");
                return Err(err);
            }
        }

        let hi = if self.token.span.is_dummy() { inner_lo } else { self.prev_token.span };

        Ok(Mod { inner: inner_lo.to(hi), items, inline: true })
    }
}

/// Reads a module from a source file.
fn eval_src_mod<'a>(
    sess: &'a ParseSess,
    cfg_mods: bool,
    path: PathBuf,
    dir_ownership: DirectoryOwnership,
    id: ast::Ident,
) -> PResult<'a, (Mod, Vec<Attribute>)> {
    let mut included_mod_stack = sess.included_mod_stack.borrow_mut();
    if let Some(i) = included_mod_stack.iter().position(|p| *p == path) {
        let mut err = String::from("circular modules: ");
        for p in &included_mod_stack[i..] {
            err.push_str(&p.to_string_lossy());
            err.push_str(" -> ");
        }
        err.push_str(&path.to_string_lossy());
        return Err(sess.span_diagnostic.struct_span_err(id.span, &err[..]));
    }
    included_mod_stack.push(path.clone());
    drop(included_mod_stack);

    let mut p0 =
        new_sub_parser_from_file(sess, &path, dir_ownership, Some(id.to_string()), id.span);
    p0.cfg_mods = cfg_mods;
    let mut module = p0.parse_mod(&token::Eof)?;
    module.0.inline = false;

    sess.included_mod_stack.borrow_mut().pop();
    Ok(module)
}

pub fn push_directory(
    id: Ident,
    attrs: &[Attribute],
    dir_ownership: &mut DirectoryOwnership,
    dir_path: &mut PathBuf,
) {
    if let Some(path) = attr::first_attr_value_str_by_name(attrs, sym::path) {
        dir_path.push(&*path.as_str());
        *dir_ownership = DirectoryOwnership::Owned { relative: None };
    } else {
        // We have to push on the current module name in the case of relative
        // paths in order to ensure that any additional module paths from inline
        // `mod x { ... }` come after the relative extension.
        //
        // For example, a `mod z { ... }` inside `x/y.rs` should set the current
        // directory path to `/x/y/z`, not `/x/z` with a relative offset of `y`.
        if let DirectoryOwnership::Owned { relative } = dir_ownership {
            if let Some(ident) = relative.take() {
                // Remove the relative offset.
                dir_path.push(&*ident.as_str());
            }
        }
        dir_path.push(&*id.as_str());
    }
}

fn submod_path<'a>(
    sess: &'a ParseSess,
    id: ast::Ident,
    attrs: &[Attribute],
    ownership: DirectoryOwnership,
    dir_path: &Path,
) -> PResult<'a, ModulePathSuccess> {
    if let Some(path) = submod_path_from_attr(attrs, dir_path) {
        let ownership = match path.file_name().and_then(|s| s.to_str()) {
            // All `#[path]` files are treated as though they are a `mod.rs` file.
            // This means that `mod foo;` declarations inside `#[path]`-included
            // files are siblings,
            //
            // Note that this will produce weirdness when a file named `foo.rs` is
            // `#[path]` included and contains a `mod foo;` declaration.
            // If you encounter this, it's your own darn fault :P
            Some(_) => DirectoryOwnership::Owned { relative: None },
            _ => DirectoryOwnership::UnownedViaMod,
        };
        return Ok(ModulePathSuccess { ownership, path });
    }

    let relative = match ownership {
        DirectoryOwnership::Owned { relative } => relative,
        DirectoryOwnership::UnownedViaBlock | DirectoryOwnership::UnownedViaMod => None,
    };
    let ModulePath { path_exists, name, result } =
        default_submod_path(sess, id, relative, dir_path);
    match ownership {
        DirectoryOwnership::Owned { .. } => Ok(result?),
        DirectoryOwnership::UnownedViaBlock => {
            let _ = result.map_err(|mut err| err.cancel());
            error_decl_mod_in_block(sess, id.span, path_exists, &name)
        }
        DirectoryOwnership::UnownedViaMod => {
            let _ = result.map_err(|mut err| err.cancel());
            error_cannot_declare_mod_here(sess, id.span, path_exists, &name)
        }
    }
}

fn error_decl_mod_in_block<'a, T>(
    sess: &'a ParseSess,
    id_sp: Span,
    path_exists: bool,
    name: &str,
) -> PResult<'a, T> {
    let msg = "Cannot declare a non-inline module inside a block unless it has a path attribute";
    let mut err = sess.span_diagnostic.struct_span_err(id_sp, msg);
    if path_exists {
        let msg = format!("Maybe `use` the module `{}` instead of redeclaring it", name);
        err.span_note(id_sp, &msg);
    }
    Err(err)
}

fn error_cannot_declare_mod_here<'a, T>(
    sess: &'a ParseSess,
    id_sp: Span,
    path_exists: bool,
    name: &str,
) -> PResult<'a, T> {
    let mut err =
        sess.span_diagnostic.struct_span_err(id_sp, "cannot declare a new module at this location");
    if !id_sp.is_dummy() {
        if let FileName::Real(src_path) = sess.source_map().span_to_filename(id_sp) {
            if let Some(stem) = src_path.file_stem() {
                let mut dest_path = src_path.clone();
                dest_path.set_file_name(stem);
                dest_path.push("mod.rs");
                err.span_note(
                    id_sp,
                    &format!(
                        "maybe move this module `{}` to its own \
                                directory via `{}`",
                        src_path.display(),
                        dest_path.display()
                    ),
                );
            }
        }
    }
    if path_exists {
        err.span_note(
            id_sp,
            &format!("... or maybe `use` the module `{}` instead of possibly redeclaring it", name),
        );
    }
    Err(err)
}

/// Derive a submodule path from the first found `#[path = "path_string"]`.
/// The provided `dir_path` is joined with the `path_string`.
// Public for rustfmt usage.
pub fn submod_path_from_attr(attrs: &[Attribute], dir_path: &Path) -> Option<PathBuf> {
    // Extract path string from first `#[path = "path_string"]` attribute.
    let path_string = attr::first_attr_value_str_by_name(attrs, sym::path)?;
    let path_string = path_string.as_str();

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
    id: ast::Ident,
    relative: Option<ast::Ident>,
    dir_path: &Path,
) -> ModulePath<'a> {
    // If we're in a foo.rs file instead of a mod.rs file,
    // we need to look for submodules in
    // `./foo/<id>.rs` and `./foo/<id>/mod.rs` rather than
    // `./<id>.rs` and `./<id>/mod.rs`.
    let relative_prefix_string;
    let relative_prefix = if let Some(ident) = relative {
        relative_prefix_string = format!("{}{}", ident.name, path::MAIN_SEPARATOR);
        &relative_prefix_string
    } else {
        ""
    };

    let mod_name = id.name.to_string();
    let default_path_str = format!("{}{}.rs", relative_prefix, mod_name);
    let secondary_path_str =
        format!("{}{}{}mod.rs", relative_prefix, mod_name, path::MAIN_SEPARATOR);
    let default_path = dir_path.join(&default_path_str);
    let secondary_path = dir_path.join(&secondary_path_str);
    let default_exists = sess.source_map().file_exists(&default_path);
    let secondary_exists = sess.source_map().file_exists(&secondary_path);

    let result = match (default_exists, secondary_exists) {
        (true, false) => Ok(ModulePathSuccess {
            path: default_path,
            ownership: DirectoryOwnership::Owned { relative: Some(id) },
        }),
        (false, true) => Ok(ModulePathSuccess {
            path: secondary_path,
            ownership: DirectoryOwnership::Owned { relative: None },
        }),
        (false, false) => {
            let mut err = struct_span_err!(
                sess.span_diagnostic,
                id.span,
                E0583,
                "file not found for module `{}`",
                mod_name,
            );
            err.help(&format!(
                "to create the module `{}`, create file \"{}\"",
                mod_name,
                default_path.display(),
            ));
            Err(err)
        }
        (true, true) => {
            let mut err = struct_span_err!(
                sess.span_diagnostic,
                id.span,
                E0584,
                "file for module `{}` found at both {} and {}",
                mod_name,
                default_path_str,
                secondary_path_str,
            );
            err.help("delete or rename one of them to remove the ambiguity");
            Err(err)
        }
    };

    ModulePath { name: mod_name, path_exists: default_exists || secondary_exists, result }
}
