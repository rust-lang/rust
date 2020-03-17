use super::diagnostics::Error;
use super::item::ItemInfo;
use super::Parser;

use crate::{new_sub_parser_from_file, DirectoryOwnership};

use rustc_ast::ast::{self, Attribute, Crate, Ident, ItemKind, Mod};
use rustc_ast::attr;
use rustc_ast::token::{self, TokenKind};
use rustc_errors::PResult;
use rustc_span::source_map::{FileName, SourceMap, Span, DUMMY_SP};
use rustc_span::symbol::sym;

use std::path::{self, Path, PathBuf};

/// Information about the path to a module.
// Public for rustfmt usage.
pub struct ModulePath {
    name: String,
    path_exists: bool,
    pub result: Result<ModulePathSuccess, Error>,
}

// Public for rustfmt usage.
pub struct ModulePathSuccess {
    pub path: PathBuf,
    pub directory_ownership: DirectoryOwnership,
}

impl<'a> Parser<'a> {
    /// Parses a source module as a crate. This is the main entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> PResult<'a, Crate> {
        let lo = self.token.span;
        let krate = Ok(ast::Crate {
            attrs: self.parse_inner_attributes()?,
            module: self.parse_mod_items(&token::Eof, lo)?,
            span: lo.to(self.token.span),
            // Filled in by proc_macro_harness::inject()
            proc_macros: Vec::new(),
        });
        krate
    }

    /// Parses a `mod <foo> { ... }` or `mod <foo>;` item.
    pub(super) fn parse_item_mod(&mut self, attrs: &mut Vec<Attribute>) -> PResult<'a, ItemInfo> {
        let in_cfg = crate::config::process_configure_mod(self.sess, self.cfg_mods, attrs);

        let id_span = self.token.span;
        let id = self.parse_ident()?;
        let (module, mut inner_attrs) = if self.eat(&token::Semi) {
            if in_cfg && self.recurse_into_file_modules {
                // This mod is in an external file. Let's go get it!
                let ModulePathSuccess { path, directory_ownership } =
                    self.submod_path(id, &attrs, id_span)?;
                self.eval_src_mod(path, directory_ownership, id.to_string(), id_span)?
            } else {
                (ast::Mod { inner: DUMMY_SP, items: Vec::new(), inline: false }, Vec::new())
            }
        } else {
            let old_directory = self.directory.clone();
            self.push_directory(id, &attrs);

            self.expect(&token::OpenDelim(token::Brace))?;
            let mod_inner_lo = self.token.span;
            let inner_attrs = self.parse_inner_attributes()?;
            let module = self.parse_mod_items(&token::CloseDelim(token::Brace), mod_inner_lo)?;

            self.directory = old_directory;
            (module, inner_attrs)
        };
        attrs.append(&mut inner_attrs);
        Ok((id, ItemKind::Mod(module)))
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

    fn submod_path(
        &mut self,
        id: ast::Ident,
        outer_attrs: &[Attribute],
        id_sp: Span,
    ) -> PResult<'a, ModulePathSuccess> {
        if let Some(path) = Parser::submod_path_from_attr(outer_attrs, &self.directory.path) {
            return Ok(ModulePathSuccess {
                directory_ownership: match path.file_name().and_then(|s| s.to_str()) {
                    // All `#[path]` files are treated as though they are a `mod.rs` file.
                    // This means that `mod foo;` declarations inside `#[path]`-included
                    // files are siblings,
                    //
                    // Note that this will produce weirdness when a file named `foo.rs` is
                    // `#[path]` included and contains a `mod foo;` declaration.
                    // If you encounter this, it's your own darn fault :P
                    Some(_) => DirectoryOwnership::Owned { relative: None },
                    _ => DirectoryOwnership::UnownedViaMod,
                },
                path,
            });
        }

        let relative = match self.directory.ownership {
            DirectoryOwnership::Owned { relative } => relative,
            DirectoryOwnership::UnownedViaBlock | DirectoryOwnership::UnownedViaMod => None,
        };
        let paths =
            Parser::default_submod_path(id, relative, &self.directory.path, self.sess.source_map());

        match self.directory.ownership {
            DirectoryOwnership::Owned { .. } => {
                paths.result.map_err(|err| self.span_fatal_err(id_sp, err))
            }
            DirectoryOwnership::UnownedViaBlock => {
                let msg = "Cannot declare a non-inline module inside a block \
                    unless it has a path attribute";
                let mut err = self.struct_span_err(id_sp, msg);
                if paths.path_exists {
                    let msg = format!(
                        "Maybe `use` the module `{}` instead of redeclaring it",
                        paths.name
                    );
                    err.span_note(id_sp, &msg);
                }
                Err(err)
            }
            DirectoryOwnership::UnownedViaMod => {
                let mut err =
                    self.struct_span_err(id_sp, "cannot declare a new module at this location");
                if !id_sp.is_dummy() {
                    let src_path = self.sess.source_map().span_to_filename(id_sp);
                    if let FileName::Real(src_path) = src_path {
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
                if paths.path_exists {
                    err.span_note(
                        id_sp,
                        &format!(
                            "... or maybe `use` the module `{}` instead \
                                            of possibly redeclaring it",
                            paths.name
                        ),
                    );
                }
                Err(err)
            }
        }
    }

    // Public for rustfmt usage.
    pub fn submod_path_from_attr(attrs: &[Attribute], dir_path: &Path) -> Option<PathBuf> {
        if let Some(s) = attr::first_attr_value_str_by_name(attrs, sym::path) {
            let s = s.as_str();

            // On windows, the base path might have the form
            // `\\?\foo\bar` in which case it does not tolerate
            // mixed `/` and `\` separators, so canonicalize
            // `/` to `\`.
            #[cfg(windows)]
            let s = s.replace("/", "\\");
            Some(dir_path.join(&*s))
        } else {
            None
        }
    }

    /// Returns a path to a module.
    // Public for rustfmt usage.
    pub fn default_submod_path(
        id: ast::Ident,
        relative: Option<ast::Ident>,
        dir_path: &Path,
        source_map: &SourceMap,
    ) -> ModulePath {
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
        let default_exists = source_map.file_exists(&default_path);
        let secondary_exists = source_map.file_exists(&secondary_path);

        let result = match (default_exists, secondary_exists) {
            (true, false) => Ok(ModulePathSuccess {
                path: default_path,
                directory_ownership: DirectoryOwnership::Owned { relative: Some(id) },
            }),
            (false, true) => Ok(ModulePathSuccess {
                path: secondary_path,
                directory_ownership: DirectoryOwnership::Owned { relative: None },
            }),
            (false, false) => {
                Err(Error::FileNotFoundForModule { mod_name: mod_name.clone(), default_path })
            }
            (true, true) => Err(Error::DuplicatePaths {
                mod_name: mod_name.clone(),
                default_path: default_path_str,
                secondary_path: secondary_path_str,
            }),
        };

        ModulePath { name: mod_name, path_exists: default_exists || secondary_exists, result }
    }

    /// Reads a module from a source file.
    fn eval_src_mod(
        &mut self,
        path: PathBuf,
        directory_ownership: DirectoryOwnership,
        name: String,
        id_sp: Span,
    ) -> PResult<'a, (Mod, Vec<Attribute>)> {
        let mut included_mod_stack = self.sess.included_mod_stack.borrow_mut();
        if let Some(i) = included_mod_stack.iter().position(|p| *p == path) {
            let mut err = String::from("circular modules: ");
            let len = included_mod_stack.len();
            for p in &included_mod_stack[i..len] {
                err.push_str(&p.to_string_lossy());
                err.push_str(" -> ");
            }
            err.push_str(&path.to_string_lossy());
            return Err(self.struct_span_err(id_sp, &err[..]));
        }
        included_mod_stack.push(path.clone());
        drop(included_mod_stack);

        let mut p0 =
            new_sub_parser_from_file(self.sess, &path, directory_ownership, Some(name), id_sp);
        p0.cfg_mods = self.cfg_mods;
        let mod_inner_lo = p0.token.span;
        let mod_attrs = p0.parse_inner_attributes()?;
        let mut m0 = p0.parse_mod_items(&token::Eof, mod_inner_lo)?;
        m0.inline = false;
        self.sess.included_mod_stack.borrow_mut().pop();
        Ok((m0, mod_attrs))
    }

    fn push_directory(&mut self, id: Ident, attrs: &[Attribute]) {
        if let Some(path) = attr::first_attr_value_str_by_name(attrs, sym::path) {
            self.directory.path.push(&*path.as_str());
            self.directory.ownership = DirectoryOwnership::Owned { relative: None };
        } else {
            // We have to push on the current module name in the case of relative
            // paths in order to ensure that any additional module paths from inline
            // `mod x { ... }` come after the relative extension.
            //
            // For example, a `mod z { ... }` inside `x/y.rs` should set the current
            // directory path to `/x/y/z`, not `/x/z` with a relative offset of `y`.
            if let DirectoryOwnership::Owned { relative } = &mut self.directory.ownership {
                if let Some(ident) = relative.take() {
                    // remove the relative offset
                    self.directory.path.push(&*ident.as_str());
                }
            }
            self.directory.path.push(&*id.as_str());
        }
    }
}
