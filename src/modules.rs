use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use syntax::ast;
use syntax::attr;
use syntax::parse::{
    new_sub_parser_from_file, parser, token, DirectoryOwnership, PResult, ParseSess,
};
use syntax::source_map::{self, Span};
use syntax::symbol::sym;
use syntax::visit::Visitor;
use syntax_pos::{self, symbol::Symbol, DUMMY_SP};

use crate::attr::MetaVisitor;
use crate::config::FileName;
use crate::items::is_mod_decl;
use crate::utils::contains_skip;

mod visitor;

type FileModMap<'ast> = BTreeMap<FileName, Cow<'ast, ast::Mod>>;

/// Maps each module to the corresponding file.
pub(crate) struct ModResolver<'ast, 'sess> {
    parse_sess: &'sess ParseSess,
    directory: Directory,
    file_map: FileModMap<'ast>,
    recursive: bool,
}

#[derive(Clone)]
struct Directory {
    path: PathBuf,
    ownership: DirectoryOwnership,
}

impl<'a> Directory {
    fn to_syntax_directory(&'a self) -> syntax::parse::Directory<'a> {
        syntax::parse::Directory {
            path: Cow::Borrowed(&self.path),
            ownership: self.ownership.clone(),
        }
    }
}

#[derive(Clone)]
enum SubModKind<'a, 'ast> {
    /// `mod foo;`
    External(PathBuf, DirectoryOwnership),
    /// `mod foo;` with multiple sources.
    MultiExternal(Vec<(PathBuf, DirectoryOwnership, Cow<'ast, ast::Mod>)>),
    /// `#[path = "..."] mod foo {}`
    InternalWithPath(PathBuf),
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
    ) -> Result<FileModMap<'ast>, String> {
        let root_filename = self.parse_sess.source_map().span_to_filename(krate.span);
        self.directory.path = match root_filename {
            source_map::FileName::Real(ref path) => path
                .parent()
                .expect("Parent directory should exists")
                .to_path_buf(),
            _ => PathBuf::new(),
        };

        // Skip visiting sub modules when the input is from stdin.
        if self.recursive {
            self.visit_mod_from_ast(&krate.module)?;
        }

        self.file_map
            .insert(root_filename.into(), Cow::Borrowed(&krate.module));
        Ok(self.file_map)
    }

    /// Visit `cfg_if` macro and look for module declarations.
    fn visit_cfg_if(&mut self, item: Cow<'ast, ast::Item>) -> Result<(), String> {
        let mut visitor =
            visitor::CfgIfVisitor::new(self.parse_sess, self.directory.to_syntax_directory());
        visitor.visit_item(&item);
        for module_item in visitor.mods() {
            if let ast::ItemKind::Mod(ref sub_mod) = module_item.item.node {
                self.visit_sub_mod(&item, Cow::Owned(sub_mod.clone()))?;
            }
        }
        Ok(())
    }

    /// Visit modules defined inside macro calls.
    fn visit_mod_outside_ast(&mut self, module: ast::Mod) -> Result<(), String> {
        for item in module.items {
            if is_cfg_if(&item) {
                self.visit_cfg_if(Cow::Owned(item.into_inner()))?;
                continue;
            }

            if let ast::ItemKind::Mod(ref sub_mod) = item.node {
                self.visit_sub_mod(&item, Cow::Owned(sub_mod.clone()))?;
            }
        }
        Ok(())
    }

    /// Visit modules from AST.
    fn visit_mod_from_ast(&mut self, module: &'ast ast::Mod) -> Result<(), String> {
        for item in &module.items {
            if is_cfg_if(item) {
                self.visit_cfg_if(Cow::Borrowed(item))?;
            }

            if let ast::ItemKind::Mod(ref sub_mod) = item.node {
                self.visit_sub_mod(item, Cow::Borrowed(sub_mod))?;
            }
        }
        Ok(())
    }

    fn visit_sub_mod(
        &mut self,
        item: &'c ast::Item,
        sub_mod: Cow<'ast, ast::Mod>,
    ) -> Result<(), String> {
        let old_directory = self.directory.clone();
        let sub_mod_kind = self.peek_sub_mod(item, &sub_mod)?;
        if let Some(sub_mod_kind) = sub_mod_kind {
            self.insert_sub_mod(sub_mod_kind.clone(), sub_mod.clone())?;
            self.visit_sub_mod_inner(sub_mod, sub_mod_kind)?;
        }
        self.directory = old_directory;
        Ok(())
    }

    /// Inspect the given sub-module which we are about to visit and returns its kind.
    fn peek_sub_mod(
        &self,
        item: &'c ast::Item,
        sub_mod: &Cow<'ast, ast::Mod>,
    ) -> Result<Option<SubModKind<'c, 'ast>>, String> {
        if contains_skip(&item.attrs) {
            return Ok(None);
        }

        if is_mod_decl(item) {
            // mod foo;
            // Look for an extern file.
            self.find_external_module(item.ident, &item.attrs, sub_mod)
                .map(Some)
        } else {
            // An internal module (`mod foo { /* ... */ }`);
            if let Some(path) = find_path_value(&item.attrs) {
                let path = Path::new(&path.as_str()).to_path_buf();
                Ok(Some(SubModKind::InternalWithPath(path)))
            } else {
                Ok(Some(SubModKind::Internal(item)))
            }
        }
    }

    fn insert_sub_mod(
        &mut self,
        sub_mod_kind: SubModKind<'c, 'ast>,
        sub_mod: Cow<'ast, ast::Mod>,
    ) -> Result<(), String> {
        match sub_mod_kind {
            SubModKind::External(mod_path, _) => {
                self.file_map.insert(FileName::Real(mod_path), sub_mod);
            }
            SubModKind::MultiExternal(mods) => {
                for (mod_path, _, sub_mod) in mods {
                    self.file_map.insert(FileName::Real(mod_path), sub_mod);
                }
            }
            _ => (),
        }
        Ok(())
    }

    fn visit_sub_mod_inner(
        &mut self,
        sub_mod: Cow<'ast, ast::Mod>,
        sub_mod_kind: SubModKind<'c, 'ast>,
    ) -> Result<(), String> {
        match sub_mod_kind {
            SubModKind::External(mod_path, directory_ownership) => {
                let directory = Directory {
                    path: mod_path.parent().unwrap().to_path_buf(),
                    ownership: directory_ownership,
                };
                self.visit_sub_mod_after_directory_update(sub_mod, Some(directory))
            }
            SubModKind::InternalWithPath(mod_path) => {
                // All `#[path]` files are treated as though they are a `mod.rs` file.
                let directory = Directory {
                    path: mod_path,
                    ownership: DirectoryOwnership::Owned { relative: None },
                };
                self.visit_sub_mod_after_directory_update(sub_mod, Some(directory))
            }
            SubModKind::Internal(ref item) => {
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
        sub_mod: Cow<'ast, ast::Mod>,
        directory: Option<Directory>,
    ) -> Result<(), String> {
        if let Some(directory) = directory {
            self.directory = directory;
        }
        match sub_mod {
            Cow::Borrowed(sub_mod) => self.visit_mod_from_ast(sub_mod),
            Cow::Owned(sub_mod) => self.visit_mod_outside_ast(sub_mod),
        }
    }

    /// Find a file path in the filesystem which corresponds to the given module.
    fn find_external_module(
        &self,
        mod_name: ast::Ident,
        attrs: &[ast::Attribute],
        sub_mod: &Cow<'ast, ast::Mod>,
    ) -> Result<SubModKind<'c, 'ast>, String> {
        if let Some(path) = parser::Parser::submod_path_from_attr(attrs, &self.directory.path) {
            return Ok(SubModKind::External(
                path,
                DirectoryOwnership::Owned { relative: None },
            ));
        }

        // Look for nested path, like `#[cfg_attr(feature = "foo", path = "bar.rs")]`.
        let mut mods_outside_ast = self
            .find_mods_ouside_of_ast(attrs, sub_mod)
            .unwrap_or(vec![]);

        let relative = match self.directory.ownership {
            DirectoryOwnership::Owned { relative } => relative,
            DirectoryOwnership::UnownedViaBlock | DirectoryOwnership::UnownedViaMod(_) => None,
        };
        match parser::Parser::default_submod_path(
            mod_name,
            relative,
            &self.directory.path,
            self.parse_sess.source_map(),
        )
        .result
        {
            Ok(parser::ModulePathSuccess {
                path,
                directory_ownership,
                ..
            }) => Ok(if mods_outside_ast.is_empty() {
                SubModKind::External(path, directory_ownership)
            } else {
                mods_outside_ast.push((path, directory_ownership, sub_mod.clone()));
                SubModKind::MultiExternal(mods_outside_ast)
            }),
            Err(_) if !mods_outside_ast.is_empty() => {
                Ok(SubModKind::MultiExternal(mods_outside_ast))
            }
            Err(_) => Err(format!(
                "Failed to find module {} in {:?} {:?}",
                mod_name, self.directory.path, relative,
            )),
        }
    }

    fn push_inline_mod_directory(&mut self, id: ast::Ident, attrs: &[ast::Attribute]) {
        if let Some(path) = find_path_value(attrs) {
            self.directory.path.push(&path.as_str());
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
            self.directory.path.push(&id.as_str());
        }
    }

    fn find_mods_ouside_of_ast(
        &self,
        attrs: &[ast::Attribute],
        sub_mod: &Cow<'ast, ast::Mod>,
    ) -> Option<Vec<(PathBuf, DirectoryOwnership, Cow<'ast, ast::Mod>)>> {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        Some(
            catch_unwind(AssertUnwindSafe(|| {
                self.find_mods_ouside_of_ast_inner(attrs, sub_mod)
            }))
            .ok()?,
        )
    }

    fn find_mods_ouside_of_ast_inner(
        &self,
        attrs: &[ast::Attribute],
        sub_mod: &Cow<'ast, ast::Mod>,
    ) -> Vec<(PathBuf, DirectoryOwnership, Cow<'ast, ast::Mod>)> {
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
            let file_name = syntax_pos::FileName::Real(actual_path.clone());
            if self
                .parse_sess
                .source_map()
                .get_source_file(&file_name)
                .is_some()
            {
                // If the specfied file is already parsed, then we just use that.
                result.push((
                    actual_path,
                    DirectoryOwnership::Owned { relative: None },
                    sub_mod.clone(),
                ));
                continue;
            }
            let mut parser = new_sub_parser_from_file(
                self.parse_sess,
                &actual_path,
                self.directory.ownership,
                None,
                DUMMY_SP,
            );
            parser.cfg_mods = false;
            let lo = parser.span;
            // FIXME(topecongiro) Format inner attributes (#3606).
            let _mod_attrs = match parse_inner_attributes(&mut parser) {
                Ok(attrs) => attrs,
                Err(mut e) => {
                    e.cancel();
                    parser.sess.span_diagnostic.reset_err_count();
                    continue;
                }
            };
            let m = match parse_mod_items(&mut parser, lo) {
                Ok(m) => m,
                Err(mut e) => {
                    e.cancel();
                    parser.sess.span_diagnostic.reset_err_count();
                    continue;
                }
            };
            result.push((
                actual_path,
                DirectoryOwnership::Owned { relative: None },
                Cow::Owned(m),
            ))
        }
        result
    }
}

fn path_value(attr: &ast::Attribute) -> Option<Symbol> {
    if attr.check_name(sym::path) {
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

// FIXME(topecongiro) Use the method from libsyntax[1] once it become public.
//
// [1] https://github.com/rust-lang/rust/blob/master/src/libsyntax/parse/attr.rs
fn parse_inner_attributes<'a>(parser: &mut parser::Parser<'a>) -> PResult<'a, Vec<ast::Attribute>> {
    let mut attrs: Vec<ast::Attribute> = vec![];
    loop {
        match parser.token {
            token::Pound => {
                // Don't even try to parse if it's not an inner attribute.
                if !parser.look_ahead(1, |t| t == &token::Not) {
                    break;
                }

                let attr = parser.parse_attribute(true)?;
                assert_eq!(attr.style, ast::AttrStyle::Inner);
                attrs.push(attr);
            }
            token::DocComment(s) => {
                // we need to get the position of this token before we bump.
                let attr = attr::mk_sugared_doc_attr(attr::mk_attr_id(), s, parser.span);
                if attr.style == ast::AttrStyle::Inner {
                    attrs.push(attr);
                    parser.bump();
                } else {
                    break;
                }
            }
            _ => break,
        }
    }
    Ok(attrs)
}

fn parse_mod_items<'a>(parser: &mut parser::Parser<'a>, inner_lo: Span) -> PResult<'a, ast::Mod> {
    let mut items = vec![];
    while let Some(item) = parser.parse_item()? {
        items.push(item);
    }

    let hi = if parser.span.is_dummy() {
        inner_lo
    } else {
        parser.prev_span
    };

    Ok(ast::Mod {
        inner: inner_lo.to(hi),
        items,
        inline: false,
    })
}

fn is_cfg_if(item: &ast::Item) -> bool {
    match item.node {
        ast::ItemKind::Mac(..) if item.ident.name == Symbol::intern("cfg_if") => true,
        _ => false,
    }
}
