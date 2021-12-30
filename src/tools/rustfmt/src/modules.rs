use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rustc_ast::ast;
use rustc_ast::visit::Visitor;
use rustc_ast::AstLike;
use rustc_span::symbol::{self, sym, Symbol};
use rustc_span::Span;
use thiserror::Error;

use crate::attr::MetaVisitor;
use crate::config::FileName;
use crate::items::is_mod_decl;
use crate::parse::parser::{
    Directory, DirectoryOwnership, ModError, ModulePathSuccess, Parser, ParserError,
};
use crate::parse::session::ParseSess;
use crate::utils::{contains_skip, mk_sp};

mod visitor;

type FileModMap<'ast> = BTreeMap<FileName, Module<'ast>>;

/// Represents module with its inner attributes.
#[derive(Debug, Clone)]
pub(crate) struct Module<'a> {
    ast_mod_kind: Option<Cow<'a, ast::ModKind>>,
    pub(crate) items: Cow<'a, Vec<rustc_ast::ptr::P<ast::Item>>>,
    inner_attr: Vec<ast::Attribute>,
    pub(crate) span: Span,
}

impl<'a> Module<'a> {
    pub(crate) fn new(
        mod_span: Span,
        ast_mod_kind: Option<Cow<'a, ast::ModKind>>,
        mod_items: Cow<'a, Vec<rustc_ast::ptr::P<ast::Item>>>,
        mod_attrs: Cow<'a, Vec<ast::Attribute>>,
    ) -> Self {
        let inner_attr = mod_attrs
            .iter()
            .filter(|attr| attr.style == ast::AttrStyle::Inner)
            .cloned()
            .collect();
        Module {
            items: mod_items,
            inner_attr,
            span: mod_span,
            ast_mod_kind,
        }
    }
}

impl<'a> AstLike for Module<'a> {
    const SUPPORTS_CUSTOM_INNER_ATTRS: bool = true;
    fn attrs(&self) -> &[ast::Attribute] {
        &self.inner_attr
    }
    fn visit_attrs(&mut self, f: impl FnOnce(&mut Vec<ast::Attribute>)) {
        f(&mut self.inner_attr)
    }
    fn tokens_mut(&mut self) -> Option<&mut Option<rustc_ast::tokenstream::LazyTokenStream>> {
        unimplemented!()
    }
}

/// Maps each module to the corresponding file.
pub(crate) struct ModResolver<'ast, 'sess> {
    parse_sess: &'sess ParseSess,
    directory: Directory,
    file_map: FileModMap<'ast>,
    recursive: bool,
}

/// Represents errors while trying to resolve modules.
#[derive(Debug, Error)]
#[error("failed to resolve mod `{module}`: {kind}")]
pub struct ModuleResolutionError {
    pub(crate) module: String,
    pub(crate) kind: ModuleResolutionErrorKind,
}

#[derive(Debug, Error)]
pub(crate) enum ModuleResolutionErrorKind {
    /// Find a file that cannot be parsed.
    #[error("cannot parse {file}")]
    ParseError { file: PathBuf },
    /// File cannot be found.
    #[error("{file} does not exist")]
    NotFound { file: PathBuf },
}

#[derive(Clone)]
enum SubModKind<'a, 'ast> {
    /// `mod foo;`
    External(PathBuf, DirectoryOwnership, Module<'ast>),
    /// `mod foo;` with multiple sources.
    MultiExternal(Vec<(PathBuf, DirectoryOwnership, Module<'ast>)>),
    /// `mod foo {}`
    Internal(&'a ast::Item),
}

impl<'ast, 'sess, 'c> ModResolver<'ast, 'sess> {
    /// Creates a new `ModResolver`.
    pub(crate) fn new(
        parse_sess: &'sess ParseSess,
        directory_ownership: DirectoryOwnership,
        recursive: bool,
    ) -> Self {
        ModResolver {
            directory: Directory {
                path: PathBuf::new(),
                ownership: directory_ownership,
            },
            file_map: BTreeMap::new(),
            parse_sess,
            recursive,
        }
    }

    /// Creates a map that maps a file name to the module in AST.
    pub(crate) fn visit_crate(
        mut self,
        krate: &'ast ast::Crate,
    ) -> Result<FileModMap<'ast>, ModuleResolutionError> {
        let root_filename = self.parse_sess.span_to_filename(krate.span);
        self.directory.path = match root_filename {
            FileName::Real(ref p) => p.parent().unwrap_or(Path::new("")).to_path_buf(),
            _ => PathBuf::new(),
        };

        // Skip visiting sub modules when the input is from stdin.
        if self.recursive {
            self.visit_mod_from_ast(&krate.items)?;
        }

        let snippet_provider = self.parse_sess.snippet_provider(krate.span);

        self.file_map.insert(
            root_filename,
            Module::new(
                mk_sp(snippet_provider.start_pos(), snippet_provider.end_pos()),
                None,
                Cow::Borrowed(&krate.items),
                Cow::Borrowed(&krate.attrs),
            ),
        );
        Ok(self.file_map)
    }

    /// Visit `cfg_if` macro and look for module declarations.
    fn visit_cfg_if(&mut self, item: Cow<'ast, ast::Item>) -> Result<(), ModuleResolutionError> {
        let mut visitor = visitor::CfgIfVisitor::new(self.parse_sess);
        visitor.visit_item(&item);
        for module_item in visitor.mods() {
            if let ast::ItemKind::Mod(_, ref sub_mod_kind) = module_item.item.kind {
                self.visit_sub_mod(
                    &module_item.item,
                    Module::new(
                        module_item.item.span,
                        Some(Cow::Owned(sub_mod_kind.clone())),
                        Cow::Owned(vec![]),
                        Cow::Owned(vec![]),
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// Visit modules defined inside macro calls.
    fn visit_mod_outside_ast(
        &mut self,
        items: Vec<rustc_ast::ptr::P<ast::Item>>,
    ) -> Result<(), ModuleResolutionError> {
        for item in items {
            if is_cfg_if(&item) {
                self.visit_cfg_if(Cow::Owned(item.into_inner()))?;
                continue;
            }

            if let ast::ItemKind::Mod(_, ref sub_mod_kind) = item.kind {
                let span = item.span;
                self.visit_sub_mod(
                    &item,
                    Module::new(
                        span,
                        Some(Cow::Owned(sub_mod_kind.clone())),
                        Cow::Owned(vec![]),
                        Cow::Owned(vec![]),
                    ),
                )?;
            }
        }
        Ok(())
    }

    /// Visit modules from AST.
    fn visit_mod_from_ast(
        &mut self,
        items: &'ast [rustc_ast::ptr::P<ast::Item>],
    ) -> Result<(), ModuleResolutionError> {
        for item in items {
            if is_cfg_if(item) {
                self.visit_cfg_if(Cow::Borrowed(item))?;
            }

            if let ast::ItemKind::Mod(_, ref sub_mod_kind) = item.kind {
                let span = item.span;
                self.visit_sub_mod(
                    item,
                    Module::new(
                        span,
                        Some(Cow::Borrowed(sub_mod_kind)),
                        Cow::Owned(vec![]),
                        Cow::Borrowed(&item.attrs),
                    ),
                )?;
            }
        }
        Ok(())
    }

    fn visit_sub_mod(
        &mut self,
        item: &'c ast::Item,
        sub_mod: Module<'ast>,
    ) -> Result<(), ModuleResolutionError> {
        let old_directory = self.directory.clone();
        let sub_mod_kind = self.peek_sub_mod(item, &sub_mod)?;
        if let Some(sub_mod_kind) = sub_mod_kind {
            self.insert_sub_mod(sub_mod_kind.clone())?;
            self.visit_sub_mod_inner(sub_mod, sub_mod_kind)?;
        }
        self.directory = old_directory;
        Ok(())
    }

    /// Inspect the given sub-module which we are about to visit and returns its kind.
    fn peek_sub_mod(
        &self,
        item: &'c ast::Item,
        sub_mod: &Module<'ast>,
    ) -> Result<Option<SubModKind<'c, 'ast>>, ModuleResolutionError> {
        if contains_skip(&item.attrs) {
            return Ok(None);
        }

        if is_mod_decl(item) {
            // mod foo;
            // Look for an extern file.
            self.find_external_module(item.ident, &item.attrs, sub_mod)
        } else {
            // An internal module (`mod foo { /* ... */ }`);
            Ok(Some(SubModKind::Internal(item)))
        }
    }

    fn insert_sub_mod(
        &mut self,
        sub_mod_kind: SubModKind<'c, 'ast>,
    ) -> Result<(), ModuleResolutionError> {
        match sub_mod_kind {
            SubModKind::External(mod_path, _, sub_mod) => {
                self.file_map
                    .entry(FileName::Real(mod_path))
                    .or_insert(sub_mod);
            }
            SubModKind::MultiExternal(mods) => {
                for (mod_path, _, sub_mod) in mods {
                    self.file_map
                        .entry(FileName::Real(mod_path))
                        .or_insert(sub_mod);
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn visit_sub_mod_inner(
        &mut self,
        sub_mod: Module<'ast>,
        sub_mod_kind: SubModKind<'c, 'ast>,
    ) -> Result<(), ModuleResolutionError> {
        match sub_mod_kind {
            SubModKind::External(mod_path, directory_ownership, sub_mod) => {
                let directory = Directory {
                    path: mod_path.parent().unwrap().to_path_buf(),
                    ownership: directory_ownership,
                };
                self.visit_sub_mod_after_directory_update(sub_mod, Some(directory))
            }
            SubModKind::Internal(item) => {
                self.push_inline_mod_directory(item.ident, &item.attrs);
                self.visit_sub_mod_after_directory_update(sub_mod, None)
            }
            SubModKind::MultiExternal(mods) => {
                for (mod_path, directory_ownership, sub_mod) in mods {
                    let directory = Directory {
                        path: mod_path.parent().unwrap().to_path_buf(),
                        ownership: directory_ownership,
                    };
                    self.visit_sub_mod_after_directory_update(sub_mod, Some(directory))?;
                }
                Ok(())
            }
        }
    }

    fn visit_sub_mod_after_directory_update(
        &mut self,
        sub_mod: Module<'ast>,
        directory: Option<Directory>,
    ) -> Result<(), ModuleResolutionError> {
        if let Some(directory) = directory {
            self.directory = directory;
        }
        match (sub_mod.ast_mod_kind, sub_mod.items) {
            (Some(Cow::Borrowed(ast::ModKind::Loaded(items, _, _))), _) => {
                self.visit_mod_from_ast(items)
            }
            (Some(Cow::Owned(ast::ModKind::Loaded(items, _, _))), _) | (_, Cow::Owned(items)) => {
                self.visit_mod_outside_ast(items)
            }
            (_, _) => Ok(()),
        }
    }

    /// Find a file path in the filesystem which corresponds to the given module.
    fn find_external_module(
        &self,
        mod_name: symbol::Ident,
        attrs: &[ast::Attribute],
        sub_mod: &Module<'ast>,
    ) -> Result<Option<SubModKind<'c, 'ast>>, ModuleResolutionError> {
        let relative = match self.directory.ownership {
            DirectoryOwnership::Owned { relative } => relative,
            DirectoryOwnership::UnownedViaBlock => None,
        };
        if let Some(path) = Parser::submod_path_from_attr(attrs, &self.directory.path) {
            if self.parse_sess.is_file_parsed(&path) {
                return Ok(None);
            }
            return match Parser::parse_file_as_module(self.parse_sess, &path, sub_mod.span) {
                Ok((ref attrs, _, _)) if contains_skip(attrs) => Ok(None),
                Ok((attrs, items, span)) => Ok(Some(SubModKind::External(
                    path,
                    DirectoryOwnership::Owned { relative: None },
                    Module::new(
                        span,
                        Some(Cow::Owned(ast::ModKind::Unloaded)),
                        Cow::Owned(items),
                        Cow::Owned(attrs),
                    ),
                ))),
                Err(ParserError::ParseError) => Err(ModuleResolutionError {
                    module: mod_name.to_string(),
                    kind: ModuleResolutionErrorKind::ParseError { file: path },
                }),
                Err(..) => Err(ModuleResolutionError {
                    module: mod_name.to_string(),
                    kind: ModuleResolutionErrorKind::NotFound { file: path },
                }),
            };
        }

        // Look for nested path, like `#[cfg_attr(feature = "foo", path = "bar.rs")]`.
        let mut mods_outside_ast = self.find_mods_outside_of_ast(attrs, sub_mod);

        match self
            .parse_sess
            .default_submod_path(mod_name, relative, &self.directory.path)
        {
            Ok(ModulePathSuccess {
                file_path,
                dir_ownership,
                ..
            }) => {
                let outside_mods_empty = mods_outside_ast.is_empty();
                let should_insert = !mods_outside_ast
                    .iter()
                    .any(|(outside_path, _, _)| outside_path == &file_path);
                if self.parse_sess.is_file_parsed(&file_path) {
                    if outside_mods_empty {
                        return Ok(None);
                    } else {
                        if should_insert {
                            mods_outside_ast.push((file_path, dir_ownership, sub_mod.clone()));
                        }
                        return Ok(Some(SubModKind::MultiExternal(mods_outside_ast)));
                    }
                }
                match Parser::parse_file_as_module(self.parse_sess, &file_path, sub_mod.span) {
                    Ok((ref attrs, _, _)) if contains_skip(attrs) => Ok(None),
                    Ok((attrs, items, span)) if outside_mods_empty => {
                        Ok(Some(SubModKind::External(
                            file_path,
                            dir_ownership,
                            Module::new(
                                span,
                                Some(Cow::Owned(ast::ModKind::Unloaded)),
                                Cow::Owned(items),
                                Cow::Owned(attrs),
                            ),
                        )))
                    }
                    Ok((attrs, items, span)) => {
                        mods_outside_ast.push((
                            file_path.clone(),
                            dir_ownership,
                            Module::new(
                                span,
                                Some(Cow::Owned(ast::ModKind::Unloaded)),
                                Cow::Owned(items),
                                Cow::Owned(attrs),
                            ),
                        ));
                        if should_insert {
                            mods_outside_ast.push((file_path, dir_ownership, sub_mod.clone()));
                        }
                        Ok(Some(SubModKind::MultiExternal(mods_outside_ast)))
                    }
                    Err(ParserError::ParseError) => Err(ModuleResolutionError {
                        module: mod_name.to_string(),
                        kind: ModuleResolutionErrorKind::ParseError { file: file_path },
                    }),
                    Err(..) if outside_mods_empty => Err(ModuleResolutionError {
                        module: mod_name.to_string(),
                        kind: ModuleResolutionErrorKind::NotFound { file: file_path },
                    }),
                    Err(..) => {
                        if should_insert {
                            mods_outside_ast.push((file_path, dir_ownership, sub_mod.clone()));
                        }
                        Ok(Some(SubModKind::MultiExternal(mods_outside_ast)))
                    }
                }
            }
            Err(mod_err) if !mods_outside_ast.is_empty() => {
                if let ModError::ParserError(mut e) = mod_err {
                    e.cancel();
                }
                Ok(Some(SubModKind::MultiExternal(mods_outside_ast)))
            }
            Err(_) => Err(ModuleResolutionError {
                module: mod_name.to_string(),
                kind: ModuleResolutionErrorKind::NotFound {
                    file: self.directory.path.clone(),
                },
            }),
        }
    }

    fn push_inline_mod_directory(&mut self, id: symbol::Ident, attrs: &[ast::Attribute]) {
        if let Some(path) = find_path_value(attrs) {
            self.directory.path.push(path.as_str());
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
                    self.directory.path.push(ident.as_str());
                }
            }
            self.directory.path.push(id.as_str());
        }
    }

    fn find_mods_outside_of_ast(
        &self,
        attrs: &[ast::Attribute],
        sub_mod: &Module<'ast>,
    ) -> Vec<(PathBuf, DirectoryOwnership, Module<'ast>)> {
        // Filter nested path, like `#[cfg_attr(feature = "foo", path = "bar.rs")]`.
        let mut path_visitor = visitor::PathVisitor::default();
        for attr in attrs.iter() {
            if let Some(meta) = attr.meta() {
                path_visitor.visit_meta_item(&meta)
            }
        }
        let mut result = vec![];
        for path in path_visitor.paths() {
            let mut actual_path = self.directory.path.clone();
            actual_path.push(&path);
            if !actual_path.exists() {
                continue;
            }
            if self.parse_sess.is_file_parsed(&actual_path) {
                // If the specified file is already parsed, then we just use that.
                result.push((
                    actual_path,
                    DirectoryOwnership::Owned { relative: None },
                    sub_mod.clone(),
                ));
                continue;
            }
            let (attrs, items, span) =
                match Parser::parse_file_as_module(self.parse_sess, &actual_path, sub_mod.span) {
                    Ok((ref attrs, _, _)) if contains_skip(attrs) => continue,
                    Ok(m) => m,
                    Err(..) => continue,
                };

            result.push((
                actual_path,
                DirectoryOwnership::Owned { relative: None },
                Module::new(
                    span,
                    Some(Cow::Owned(ast::ModKind::Unloaded)),
                    Cow::Owned(items),
                    Cow::Owned(attrs),
                ),
            ))
        }
        result
    }
}

fn path_value(attr: &ast::Attribute) -> Option<Symbol> {
    if attr.has_name(sym::path) {
        attr.value_str()
    } else {
        None
    }
}

// N.B., even when there are multiple `#[path = ...]` attributes, we just need to
// examine the first one, since rustc ignores the second and the subsequent ones
// as unused attributes.
fn find_path_value(attrs: &[ast::Attribute]) -> Option<Symbol> {
    attrs.iter().flat_map(path_value).next()
}

fn is_cfg_if(item: &ast::Item) -> bool {
    match item.kind {
        ast::ItemKind::MacCall(ref mac) => {
            if let Some(first_segment) = mac.path.segments.first() {
                if first_segment.ident.name == Symbol::intern("cfg_if") {
                    return true;
                }
            }
            false
        }
        _ => false,
    }
}
