// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

use syntax::ast;
use syntax::parse::{parser, DirectoryOwnership};
use syntax::source_map;
use syntax_pos::symbol::Symbol;

use config::FileName;
use utils::contains_skip;

/// List all the files containing modules of a crate.
/// If a file is used twice in a crate, it appears only once.
pub fn list_files<'a>(
    krate: &'a ast::Crate,
    source_map: &source_map::SourceMap,
) -> Result<BTreeMap<FileName, &'a ast::Mod>, io::Error> {
    let mut result = BTreeMap::new(); // Enforce file order determinism
    let root_filename = source_map.span_to_filename(krate.span);
    {
        let parent = match root_filename {
            source_map::FileName::Real(ref path) => path.parent().unwrap(),
            _ => Path::new(""),
        };
        list_submodules(&krate.module, parent, None, source_map, &mut result)?;
    }
    result.insert(root_filename.into(), &krate.module);
    Ok(result)
}

fn path_value(attr: &ast::Attribute) -> Option<Symbol> {
    if attr.name() == "path" {
        attr.value_str()
    } else {
        None
    }
}

// N.B. Even when there are multiple `#[path = ...]` attributes, we just need to
// examine the first one, since rustc ignores the second and the subsequent ones
// as unused attributes.
fn find_path_value(attrs: &[ast::Attribute]) -> Option<Symbol> {
    attrs.iter().flat_map(path_value).next()
}

/// Recursively list all external modules included in a module.
fn list_submodules<'a>(
    module: &'a ast::Mod,
    search_dir: &Path,
    relative: Option<ast::Ident>,
    source_map: &source_map::SourceMap,
    result: &mut BTreeMap<FileName, &'a ast::Mod>,
) -> Result<(), io::Error> {
    debug!("list_submodules: search_dir: {:?}", search_dir);
    for item in &module.items {
        if let ast::ItemKind::Mod(ref sub_mod) = item.node {
            if !contains_skip(&item.attrs) {
                let is_internal = source_map.span_to_filename(item.span)
                    == source_map.span_to_filename(sub_mod.inner);
                let (dir_path, relative) = if is_internal {
                    if let Some(path) = find_path_value(&item.attrs) {
                        (search_dir.join(&path.as_str()), None)
                    } else {
                        (search_dir.join(&item.ident.to_string()), None)
                    }
                } else {
                    let (mod_path, relative) =
                        module_file(item.ident, &item.attrs, search_dir, relative, source_map)?;
                    let dir_path = mod_path.parent().unwrap().to_owned();
                    result.insert(FileName::Real(mod_path), sub_mod);
                    (dir_path, relative)
                };
                list_submodules(sub_mod, &dir_path, relative, source_map, result)?;
            }
        }
    }
    Ok(())
}

/// Find the file corresponding to an external mod
fn module_file(
    id: ast::Ident,
    attrs: &[ast::Attribute],
    dir_path: &Path,
    relative: Option<ast::Ident>,
    source_map: &source_map::SourceMap,
) -> Result<(PathBuf, Option<ast::Ident>), io::Error> {
    if let Some(path) = parser::Parser::submod_path_from_attr(attrs, dir_path) {
        return Ok((path, None));
    }

    match parser::Parser::default_submod_path(id, relative, dir_path, source_map).result {
        Ok(parser::ModulePathSuccess {
            path,
            directory_ownership,
            ..
        }) => {
            let relative = if let DirectoryOwnership::Owned { relative } = directory_ownership {
                relative
            } else {
                None
            };
            Ok((path, relative))
        }
        Err(_) => Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Couldn't find module {}", id),
        )),
    }
}
