// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that checks to make sure private fields and methods aren't used
//! outside their scopes. This pass will also generate a set of exported items
//! which are available for use externally when compiled as a library.

pub use self::PrivateDep::*;
pub use self::ImportUse::*;
pub use self::LastPrivate::*;

use util::nodemap::{DefIdSet, NodeSet};

use syntax::ast;

/// A set of AST nodes exported by the crate.
pub type ExportedItems = NodeSet;

/// A set containing all exported definitions from external crates.
/// The set does not contain any entries from local crates.
pub type ExternalExports = DefIdSet;

/// A set of AST nodes that are fully public in the crate. This map is used for
/// documentation purposes (reexporting a private struct inlines the doc,
/// reexporting a public struct doesn't inline the doc).
pub type PublicItems = NodeSet;

#[derive(Copy, Clone, Debug)]
pub enum LastPrivate {
    LastMod(PrivateDep),
    // `use` directives (imports) can refer to two separate definitions in the
    // type and value namespaces. We record here the last private node for each
    // and whether the import is in fact used for each.
    // If the Option<PrivateDep> fields are None, it means there is no definition
    // in that namespace.
    LastImport{value_priv: Option<PrivateDep>,
               value_used: ImportUse,
               type_priv: Option<PrivateDep>,
               type_used: ImportUse},
}

#[derive(Copy, Clone, Debug)]
pub enum PrivateDep {
    AllPublic,
    DependsOn(ast::DefId),
}

// How an import is used.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum ImportUse {
    Unused,       // The import is not used.
    Used,         // The import is used.
}

impl LastPrivate {
    pub fn or(self, other: LastPrivate) -> LastPrivate {
        match (self, other) {
            (me, LastMod(AllPublic)) => me,
            (_, other) => other,
        }
    }
}
