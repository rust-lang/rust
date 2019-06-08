use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use syntax::ast;
use syntax::parse::{parser, DirectoryOwnership, ParseSess};
use syntax::source_map;
use syntax::symbol::sym;
use syntax::visit::Visitor;
use syntax_pos::symbol::Symbol;

use crate::config::FileName;
use crate::items::is_mod_decl;
use crate::utils::contains_skip;

mod visitor;

type FileModMap<'ast> = BTreeMap<FileName, (Cow<'ast, ast::Mod>, String)>;

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

enum SubModKind {
    /// `mod foo;`
    External(PathBuf, DirectoryOwnership),
    /// `#[path = "..."] mod foo {}`
    InternalWithPath(PathBuf),
    /// `mod foo {}`
    Internal,
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

        self.file_map.insert(
            root_filename.into(),
            (Cow::Borrowed(&krate.module), String::new()),
        );
        Ok(self.file_map)
    }

    /// Visit macro calls and look for module declarations. Currently only supports `cfg_if` macro.
    fn visit_mac(&mut self, item: Cow<'ast, ast::Item>) -> Result<(), String> {
        let mut visitor =
            visitor::CfgIfVisitor::new(self.parse_sess, self.directory.to_syntax_directory());
        visitor.visit_item(&item);
        for module_item in visitor.mods() {
            if let ast::ItemKind::Mod(ref sub_mod) = module_item.item.node {
                self.visit_mod_from_mac_inner(&item, Cow::Owned(sub_mod.clone()))?;
            }
        }
        Ok(())
    }

    /// Visit modules defined inside macro calls.
    fn visit_mod_from_macro(&mut self, module: Cow<'ast, ast::Mod>) -> Result<(), String> {
        for item in &module.items {
            if let ast::ItemKind::Mac(..) = item.node {
                self.visit_mac(Cow::Owned(item.clone().into_inner()))?;
            }

            if let ast::ItemKind::Mod(ref sub_mod) = item.node {
                self.visit_mod_from_mac_inner(item, Cow::Owned(sub_mod.clone()))?;
            }
        }
        Ok(())
    }

    fn visit_mod_from_mac_inner(
        &mut self,
        item: &'c ast::Item,
        sub_mod: Cow<'ast, ast::Mod>,
    ) -> Result<(), String> {
        let old_directory = self.directory.clone();
        self.visit_sub_mod(item, &sub_mod)?;
        self.visit_mod_from_macro(sub_mod)?;
        self.directory = old_directory;
        Ok(())
    }

    /// Visit modules from AST.
    fn visit_mod_from_ast(&mut self, module: &'ast ast::Mod) -> Result<(), String> {
        for item in &module.items {
            if let ast::ItemKind::Mac(..) = item.node {
                self.visit_mac(Cow::Borrowed(item))?;
            }

            if let ast::ItemKind::Mod(ref sub_mod) = item.node {
                let old_directory = self.directory.clone();
                self.visit_sub_mod(item, &Cow::Borrowed(sub_mod))?;
                self.visit_mod_from_ast(sub_mod)?;
                self.directory = old_directory;
            }
        }
        Ok(())
    }

    fn visit_sub_mod(
        &mut self,
        item: &'c ast::Item,
        sub_mod: &Cow<'ast, ast::Mod>,
    ) -> Result<(), String> {
        match self.peek_sub_mod(item)? {
            Some(SubModKind::External(mod_path, directory_ownership)) => {
                self.file_map.insert(
                    FileName::Real(mod_path.clone()),
                    (sub_mod.clone(), item.ident.name.as_str().get().to_owned()),
                );
                self.directory = Directory {
                    path: mod_path.parent().unwrap().to_path_buf(),
                    ownership: directory_ownership,
                };
            }
            Some(SubModKind::InternalWithPath(mod_path)) => {
                // All `#[path]` files are treated as though they are a `mod.rs` file.
                self.directory = Directory {
                    path: mod_path,
                    ownership: DirectoryOwnership::Owned { relative: None },
                };
            }
            Some(SubModKind::Internal) => self.push_inline_mod_directory(item.ident, &item.attrs),
            None => (), // rustfmt::skip
        }
        Ok(())
    }

    /// Inspect the given sub-module which we are about to visit and returns its kind.
    fn peek_sub_mod(&self, item: &'c ast::Item) -> Result<Option<SubModKind>, String> {
        if contains_skip(&item.attrs) {
            return Ok(None);
        }

        if is_mod_decl(item) {
            // mod foo;
            // Look for an extern file.
            let (mod_path, directory_ownership) =
                self.find_external_module(item.ident, &item.attrs)?;
            Ok(Some(SubModKind::External(mod_path, directory_ownership)))
        } else {
            // An internal module (`mod foo { /* ... */ }`);
            if let Some(path) = find_path_value(&item.attrs) {
                let path = Path::new(&path.as_str()).to_path_buf();
                Ok(Some(SubModKind::InternalWithPath(path)))
            } else {
                Ok(Some(SubModKind::Internal))
            }
        }
    }

    /// Find a file path in the filesystem which corresponds to the given module.
    fn find_external_module(
        &self,
        mod_name: ast::Ident,
        attrs: &[ast::Attribute],
    ) -> Result<(PathBuf, DirectoryOwnership), String> {
        if let Some(path) = parser::Parser::submod_path_from_attr(attrs, &self.directory.path) {
            return Ok((path, DirectoryOwnership::Owned { relative: None }));
        }

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
            }) => Ok((path, directory_ownership)),
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
