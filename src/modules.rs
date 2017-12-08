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
use std::path::{Path, PathBuf};
use std::io;

use syntax::ast;
use syntax::codemap::{self, FileName};
use syntax::parse::parser;

use utils::contains_skip;

/// List all the files containing modules of a crate.
/// If a file is used twice in a crate, it appears only once.
pub fn list_files<'a>(
    krate: &'a ast::Crate,
    codemap: &codemap::CodeMap,
) -> Result<BTreeMap<FileName, &'a ast::Mod>, io::Error> {
    let mut result = BTreeMap::new(); // Enforce file order determinism
    let root_filename = codemap.span_to_filename(krate.span);
    {
        let parent = match root_filename {
            FileName::Real(ref path) => path.parent().unwrap(),
            _ => Path::new(""),
        };
        list_submodules(&krate.module, parent, codemap, &mut result)?;
    }
    result.insert(root_filename, &krate.module);
    Ok(result)
}

/// Recursively list all external modules included in a module.
fn list_submodules<'a>(
    module: &'a ast::Mod,
    search_dir: &Path,
    codemap: &codemap::CodeMap,
    result: &mut BTreeMap<FileName, &'a ast::Mod>,
) -> Result<(), io::Error> {
    debug!("list_submodules: search_dir: {:?}", search_dir);
    for item in &module.items {
        if let ast::ItemKind::Mod(ref sub_mod) = item.node {
            if !contains_skip(&item.attrs) {
                let is_internal =
                    codemap.span_to_filename(item.span) == codemap.span_to_filename(sub_mod.inner);
                let dir_path = if is_internal {
                    search_dir.join(&item.ident.to_string())
                } else {
                    let mod_path = module_file(item.ident, &item.attrs, search_dir, codemap)?;
                    let dir_path = mod_path.parent().unwrap().to_owned();
                    result.insert(FileName::Real(mod_path), sub_mod);
                    dir_path
                };
                list_submodules(sub_mod, &dir_path, codemap, result)?;
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
    codemap: &codemap::CodeMap,
) -> Result<PathBuf, io::Error> {
    if let Some(path) = parser::Parser::submod_path_from_attr(attrs, dir_path) {
        return Ok(path);
    }

    match parser::Parser::default_submod_path(id, dir_path, codemap).result {
        Ok(parser::ModulePathSuccess { path, .. }) => Ok(path),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::Other,
            format!("Couldn't find module {}", id),
        )),
    }
}
