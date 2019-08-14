use super::{Parser, PResult};
use super::item::ItemInfo;

use crate::attr;
use crate::ast::{self, Ident, Attribute, ItemKind, Mod, Crate};
use crate::parse::{new_sub_parser_from_file, DirectoryOwnership};
use crate::parse::token::{self, TokenKind};
use crate::parse::diagnostics::{Error};
use crate::source_map::{SourceMap, Span, DUMMY_SP, FileName};
use crate::symbol::sym;

use std::path::{self, Path, PathBuf};

/// Information about the path to a module.
pub struct ModulePath {
    name: String,
    path_exists: bool,
    pub result: Result<ModulePathSuccess, Error>,
}

pub struct ModulePathSuccess {
    pub path: PathBuf,
    pub directory_ownership: DirectoryOwnership,
    warn: bool,
}

impl<'a> Parser<'a> {
    /// Parses a source module as a crate. This is the main entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> PResult<'a, Crate> {
        let lo = self.token.span;
        let krate = Ok(ast::Crate {
            attrs: self.parse_inner_attributes()?,
            module: self.parse_mod_items(&token::Eof, lo)?,
            span: lo.to(self.token.span),
        });
        krate
    }

    /// Parse a `mod <foo> { ... }` or `mod <foo>;` item
    pub(super) fn parse_item_mod(&mut self, outer_attrs: &[Attribute]) -> PResult<'a, ItemInfo> {
        let (in_cfg, outer_attrs) = {
            let mut strip_unconfigured = crate::config::StripUnconfigured {
                sess: self.sess,
                features: None, // don't perform gated feature checking
            };
            let mut outer_attrs = outer_attrs.to_owned();
            strip_unconfigured.process_cfg_attrs(&mut outer_attrs);
            (!self.cfg_mods || strip_unconfigured.in_cfg(&outer_attrs), outer_attrs)
        };

        let id_span = self.token.span;
        let id = self.parse_ident()?;
        if self.eat(&token::Semi) {
            if in_cfg && self.recurse_into_file_modules {
                // This mod is in an external file. Let's go get it!
                let ModulePathSuccess { path, directory_ownership, warn } =
                    self.submod_path(id, &outer_attrs, id_span)?;
                let (module, mut attrs) =
                    self.eval_src_mod(path, directory_ownership, id.to_string(), id_span)?;
                // Record that we fetched the mod from an external file
                if warn {
                    let attr = attr::mk_attr_outer(
                        attr::mk_word_item(Ident::with_empty_ctxt(sym::warn_directory_ownership)));
                    attr::mark_known(&attr);
                    attrs.push(attr);
                }
                Ok((id, ItemKind::Mod(module), Some(attrs)))
            } else {
                let placeholder = ast::Mod {
                    inner: DUMMY_SP,
                    items: Vec::new(),
                    inline: false
                };
                Ok((id, ItemKind::Mod(placeholder), None))
            }
        } else {
            let old_directory = self.directory.clone();
            self.push_directory(id, &outer_attrs);

            self.expect(&token::OpenDelim(token::Brace))?;
            let mod_inner_lo = self.token.span;
            let attrs = self.parse_inner_attributes()?;
            let module = self.parse_mod_items(&token::CloseDelim(token::Brace), mod_inner_lo)?;

            self.directory = old_directory;
            Ok((id, ItemKind::Mod(module), Some(attrs)))
        }
    }

    /// Given a termination token, parses all of the items in a module.
    fn parse_mod_items(&mut self, term: &TokenKind, inner_lo: Span) -> PResult<'a, Mod> {
        let mut items = vec![];
        while let Some(item) = self.parse_item()? {
            items.push(item);
            self.maybe_consume_incorrect_semicolon(&items);
        }

        if !self.eat(term) {
            let token_str = self.this_token_descr();
            if !self.maybe_consume_incorrect_semicolon(&items) {
                let mut err = self.fatal(&format!("expected item, found {}", token_str));
                err.span_label(self.token.span, "expected item");
                return Err(err);
            }
        }

        let hi = if self.token.span.is_dummy() {
            inner_lo
        } else {
            self.prev_span
        };

        Ok(Mod {
            inner: inner_lo.to(hi),
            items,
            inline: true
        })
    }

    fn submod_path(
        &mut self,
        id: ast::Ident,
        outer_attrs: &[Attribute],
        id_sp: Span
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
                    _ => DirectoryOwnership::UnownedViaMod(true),
                },
                path,
                warn: false,
            });
        }

        let relative = match self.directory.ownership {
            DirectoryOwnership::Owned { relative } => relative,
            DirectoryOwnership::UnownedViaBlock |
            DirectoryOwnership::UnownedViaMod(_) => None,
        };
        let paths = Parser::default_submod_path(
                        id, relative, &self.directory.path, self.sess.source_map());

        match self.directory.ownership {
            DirectoryOwnership::Owned { .. } => {
                paths.result.map_err(|err| self.span_fatal_err(id_sp, err))
            },
            DirectoryOwnership::UnownedViaBlock => {
                let msg =
                    "Cannot declare a non-inline module inside a block \
                    unless it has a path attribute";
                let mut err = self.diagnostic().struct_span_err(id_sp, msg);
                if paths.path_exists {
                    let msg = format!("Maybe `use` the module `{}` instead of redeclaring it",
                                      paths.name);
                    err.span_note(id_sp, &msg);
                }
                Err(err)
            }
            DirectoryOwnership::UnownedViaMod(warn) => {
                if warn {
                    if let Ok(result) = paths.result {
                        return Ok(ModulePathSuccess { warn: true, ..result });
                    }
                }
                let mut err = self.diagnostic().struct_span_err(id_sp,
                    "cannot declare a new module at this location");
                if !id_sp.is_dummy() {
                    let src_path = self.sess.source_map().span_to_filename(id_sp);
                    if let FileName::Real(src_path) = src_path {
                        if let Some(stem) = src_path.file_stem() {
                            let mut dest_path = src_path.clone();
                            dest_path.set_file_name(stem);
                            dest_path.push("mod.rs");
                            err.span_note(id_sp,
                                    &format!("maybe move this module `{}` to its own \
                                                directory via `{}`", src_path.display(),
                                            dest_path.display()));
                        }
                    }
                }
                if paths.path_exists {
                    err.span_note(id_sp,
                                  &format!("... or maybe `use` the module `{}` instead \
                                            of possibly redeclaring it",
                                           paths.name));
                }
                Err(err)
            }
        }
    }

    pub fn submod_path_from_attr(attrs: &[Attribute], dir_path: &Path) -> Option<PathBuf> {
        if let Some(s) = attr::first_attr_value_str_by_name(attrs, sym::path) {
            let s = s.as_str();

            // On windows, the base path might have the form
            // `\\?\foo\bar` in which case it does not tolerate
            // mixed `/` and `\` separators, so canonicalize
            // `/` to `\`.
            #[cfg(windows)]
            let s = s.replace("/", "\\");
            Some(dir_path.join(s))
        } else {
            None
        }
    }

    /// Returns a path to a module.
    pub fn default_submod_path(
        id: ast::Ident,
        relative: Option<ast::Ident>,
        dir_path: &Path,
        source_map: &SourceMap) -> ModulePath
    {
        // If we're in a foo.rs file instead of a mod.rs file,
        // we need to look for submodules in
        // `./foo/<id>.rs` and `./foo/<id>/mod.rs` rather than
        // `./<id>.rs` and `./<id>/mod.rs`.
        let relative_prefix_string;
        let relative_prefix = if let Some(ident) = relative {
            relative_prefix_string = format!("{}{}", ident.as_str(), path::MAIN_SEPARATOR);
            &relative_prefix_string
        } else {
            ""
        };

        let mod_name = id.to_string();
        let default_path_str = format!("{}{}.rs", relative_prefix, mod_name);
        let secondary_path_str = format!("{}{}{}mod.rs",
                                         relative_prefix, mod_name, path::MAIN_SEPARATOR);
        let default_path = dir_path.join(&default_path_str);
        let secondary_path = dir_path.join(&secondary_path_str);
        let default_exists = source_map.file_exists(&default_path);
        let secondary_exists = source_map.file_exists(&secondary_path);

        let result = match (default_exists, secondary_exists) {
            (true, false) => Ok(ModulePathSuccess {
                path: default_path,
                directory_ownership: DirectoryOwnership::Owned {
                    relative: Some(id),
                },
                warn: false,
            }),
            (false, true) => Ok(ModulePathSuccess {
                path: secondary_path,
                directory_ownership: DirectoryOwnership::Owned {
                    relative: None,
                },
                warn: false,
            }),
            (false, false) => Err(Error::FileNotFoundForModule {
                mod_name: mod_name.clone(),
                default_path: default_path_str,
                secondary_path: secondary_path_str,
                dir_path: dir_path.display().to_string(),
            }),
            (true, true) => Err(Error::DuplicatePaths {
                mod_name: mod_name.clone(),
                default_path: default_path_str,
                secondary_path: secondary_path_str,
            }),
        };

        ModulePath {
            name: mod_name,
            path_exists: default_exists || secondary_exists,
            result,
        }
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
            for p in &included_mod_stack[i.. len] {
                err.push_str(&p.to_string_lossy());
                err.push_str(" -> ");
            }
            err.push_str(&path.to_string_lossy());
            return Err(self.span_fatal(id_sp, &err[..]));
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
            self.directory.path.to_mut().push(&path.as_str());
            self.directory.ownership = DirectoryOwnership::Owned { relative: None };
        } else {
            // We have to push on the current module name in the case of relative
            // paths in order to ensure that any additional module paths from inline
            // `mod x { ... }` come after the relative extension.
            //
            // For example, a `mod z { ... }` inside `x/y.rs` should set the current
            // directory path to `/x/y/z`, not `/x/z` with a relative offset of `y`.
            if let DirectoryOwnership::Owned { relative } = &mut self.directory.ownership {
                if let Some(ident) = relative.take() { // remove the relative offset
                    self.directory.path.to_mut().push(ident.as_str());
                }
            }
            self.directory.path.to_mut().push(&id.as_str());
        }
    }
}
