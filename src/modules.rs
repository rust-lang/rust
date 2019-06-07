use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use syntax::ast;
use syntax::parse::{parser, DirectoryOwnership};
use syntax::source_map;
use syntax_pos::symbol::Symbol;

use crate::config::FileName;
use crate::items::is_mod_decl;
use crate::utils::contains_skip;

type FileModMap<'a> = BTreeMap<FileName, (&'a ast::Mod, &'a str)>;

/// Maps each module to the corresponding file.
pub(crate) struct ModResolver<'a, 'b> {
    source_map: &'b source_map::SourceMap,
    directory: Directory,
    file_map: FileModMap<'a>,
    recursive: bool,
}

#[derive(Clone)]
struct Directory {
    path: PathBuf,
    ownership: DirectoryOwnership,
}

impl<'a, 'b> ModResolver<'a, 'b> {
    /// Creates a new `ModResolver`.
    pub(crate) fn new(
        source_map: &'b source_map::SourceMap,
        directory_ownership: DirectoryOwnership,
        recursive: bool,
    ) -> Self {
        ModResolver {
            directory: Directory {
                path: PathBuf::new(),
                ownership: directory_ownership,
            },
            file_map: BTreeMap::new(),
            source_map,
            recursive,
        }
    }

    /// Creates a map that maps a file name to the module in AST.
    pub(crate) fn visit_crate(mut self, krate: &'a ast::Crate) -> Result<FileModMap<'a>, String> {
        let root_filename = self.source_map.span_to_filename(krate.span);
        self.directory.path = match root_filename {
            source_map::FileName::Real(ref path) => path
                .parent()
                .expect("Parent directory should exists")
                .to_path_buf(),
            _ => PathBuf::new(),
        };

        // Skip visiting sub modules when the input is from stdin.
        if self.recursive {
            self.visit_mod(&krate.module)?;
        }

        self.file_map
            .insert(root_filename.into(), (&krate.module, ""));
        Ok(self.file_map)
    }

    fn visit_mod(&mut self, module: &'a ast::Mod) -> Result<(), String> {
        for item in &module.items {
            if let ast::ItemKind::Mod(ref sub_mod) = item.node {
                if contains_skip(&item.attrs) {
                    continue;
                }

                let old_direcotry = self.directory.clone();
                if is_mod_decl(item) {
                    // mod foo;
                    // Look for an extern file.
                    let (mod_path, directory_ownership) =
                        self.find_external_module(item.ident, &item.attrs)?;
                    self.file_map.insert(
                        FileName::Real(mod_path.clone()),
                        (sub_mod, item.ident.name.as_str().get()),
                    );
                    self.directory = Directory {
                        path: mod_path.parent().unwrap().to_path_buf(),
                        ownership: directory_ownership,
                    }
                } else {
                    // An internal module (`mod foo { /* ... */ }`);
                    if let Some(path) = find_path_value(&item.attrs) {
                        // All `#[path]` files are treated as though they are a `mod.rs` file.
                        self.directory = Directory {
                            path: Path::new(&path.as_str()).to_path_buf(),
                            ownership: DirectoryOwnership::Owned { relative: None },
                        };
                    } else {
                        self.push_inline_mod_directory(item.ident, &item.attrs);
                    }
                }
                self.visit_mod(sub_mod)?;
                self.directory = old_direcotry;
            }
        }
        Ok(())
    }

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
            self.source_map,
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
    if attr.name() == "path" {
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
