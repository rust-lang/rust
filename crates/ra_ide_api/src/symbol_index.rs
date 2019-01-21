//! This module handles fuzzy-searching of functions, structs and other symbols
//! by name across the whole workspace and dependencies.
//!
//! It works by building an incrementally-updated text-search index of all
//! symbols. The backbone of the index is the **awesome** `fst` crate by
//! @BurntSushi.
//!
//! In a nutshell, you give a set of strings to the `fst`, and it builds a
//! finite state machine describing this set of strings. The strings which
//! could fuzzy-match a pattern can also be described by a finite state machine.
//! What is freakingly cool is that you can now traverse both state machines in
//! lock-step to enumerate the strings which are both in the input set and
//! fuzz-match the query. Or, more formally, given two languages described by
//! fsts, one can build an product fst which describes the intersection of the
//! languages.
//!
//! `fst` does not support cheap updating of the index, but it supports unioning
//! of state machines. So, to account for changing source code, we build an fst
//! for each library (which is assumed to never change) and an fst for each rust
//! file in the current workspace, and run a query against the union of all
//! those fsts.
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    sync::Arc,
};

use fst::{self, Streamer};
use ra_syntax::{
    SyntaxNode, SourceFile, SmolStr, TreeArc, AstNode,
    algo::{visit::{visitor, Visitor}, find_covering_node},
    SyntaxKind::{self, *},
    ast::{self, NameOwner},
};
use ra_db::{
    SourceRootId, FilesDatabase, LocalSyntaxPtr,
    salsa::{self, ParallelDatabase},
};
use rayon::prelude::*;

use crate::{
    FileId, Query,
    db::RootDatabase,
};

#[salsa::query_group]
pub(crate) trait SymbolsDatabase: hir::db::HirDatabase {
    fn file_symbols(&self, file_id: FileId) -> Arc<SymbolIndex>;
    #[salsa::input]
    fn library_symbols(&self, id: SourceRootId) -> Arc<SymbolIndex>;
}

fn file_symbols(db: &impl SymbolsDatabase, file_id: FileId) -> Arc<SymbolIndex> {
    db.check_canceled();
    let source_file = db.source_file(file_id);
    let mut symbols = source_file
        .syntax()
        .descendants()
        .filter_map(to_symbol)
        .map(move |(name, ptr)| FileSymbol { name, ptr, file_id })
        .collect::<Vec<_>>();

    for (name, text_range) in hir::source_binder::macro_symbols(db, file_id) {
        let node = find_covering_node(source_file.syntax(), text_range);
        let ptr = LocalSyntaxPtr::new(node);
        symbols.push(FileSymbol { file_id, name, ptr })
    }

    Arc::new(SymbolIndex::new(symbols))
}

pub(crate) fn world_symbols(db: &RootDatabase, query: Query) -> Vec<FileSymbol> {
    /// Need to wrap Snapshot to provide `Clone` impl for `map_with`
    struct Snap(salsa::Snapshot<RootDatabase>);
    impl Clone for Snap {
        fn clone(&self) -> Snap {
            Snap(self.0.snapshot())
        }
    }

    let buf: Vec<Arc<SymbolIndex>> = if query.libs {
        let snap = Snap(db.snapshot());
        db.library_roots()
            .par_iter()
            .map_with(snap, |db, &lib_id| db.0.library_symbols(lib_id))
            .collect()
    } else {
        let mut files = Vec::new();
        for &root in db.local_roots().iter() {
            let sr = db.source_root(root);
            files.extend(sr.files.values().map(|&it| it))
        }

        let snap = Snap(db.snapshot());
        files
            .par_iter()
            .map_with(snap, |db, &file_id| db.0.file_symbols(file_id))
            .collect()
    };
    query.search(&buf)
}

#[derive(Default, Debug)]
pub(crate) struct SymbolIndex {
    symbols: Vec<FileSymbol>,
    map: fst::Map,
}

impl PartialEq for SymbolIndex {
    fn eq(&self, other: &SymbolIndex) -> bool {
        self.symbols == other.symbols
    }
}

impl Eq for SymbolIndex {}

impl Hash for SymbolIndex {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.symbols.hash(hasher)
    }
}

impl SymbolIndex {
    fn new(mut symbols: Vec<FileSymbol>) -> SymbolIndex {
        fn cmp(s1: &FileSymbol, s2: &FileSymbol) -> Ordering {
            unicase::Ascii::new(s1.name.as_str()).cmp(&unicase::Ascii::new(s2.name.as_str()))
        }
        symbols.par_sort_by(cmp);
        symbols.dedup_by(|s1, s2| cmp(s1, s2) == Ordering::Equal);
        let names = symbols.iter().map(|it| it.name.as_str().to_lowercase());
        let map = fst::Map::from_iter(names.into_iter().zip(0u64..)).unwrap();
        SymbolIndex { symbols, map }
    }

    pub(crate) fn len(&self) -> usize {
        self.symbols.len()
    }

    pub(crate) fn for_files(
        files: impl ParallelIterator<Item = (FileId, TreeArc<SourceFile>)>,
    ) -> SymbolIndex {
        let symbols = files
            .flat_map(|(file_id, file)| {
                file.syntax()
                    .descendants()
                    .filter_map(to_symbol)
                    .map(move |(name, ptr)| FileSymbol { name, ptr, file_id })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        SymbolIndex::new(symbols)
    }
}

impl Query {
    pub(crate) fn search(self, indices: &[Arc<SymbolIndex>]) -> Vec<FileSymbol> {
        let mut op = fst::map::OpBuilder::new();
        for file_symbols in indices.iter() {
            let automaton = fst::automaton::Subsequence::new(&self.lowercased);
            op = op.add(file_symbols.map.search(automaton))
        }
        let mut stream = op.union();
        let mut res = Vec::new();
        while let Some((_, indexed_values)) = stream.next() {
            if res.len() >= self.limit {
                break;
            }
            for indexed_value in indexed_values {
                let file_symbols = &indices[indexed_value.index];
                let idx = indexed_value.value as usize;

                let symbol = &file_symbols.symbols[idx];
                if self.only_types && !is_type(symbol.ptr.kind()) {
                    continue;
                }
                if self.exact && symbol.name != self.query {
                    continue;
                }
                res.push(symbol.clone());
            }
        }
        res
    }
}

fn is_type(kind: SyntaxKind) -> bool {
    match kind {
        STRUCT_DEF | ENUM_DEF | TRAIT_DEF | TYPE_DEF => true,
        _ => false,
    }
}

/// The actual data that is stored in the index. It should be as compact as
/// possible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FileSymbol {
    pub(crate) file_id: FileId,
    pub(crate) name: SmolStr,
    pub(crate) ptr: LocalSyntaxPtr,
}

fn to_symbol(node: &SyntaxNode) -> Option<(SmolStr, LocalSyntaxPtr)> {
    fn decl<N: NameOwner>(node: &N) -> Option<(SmolStr, LocalSyntaxPtr)> {
        let name = node.name()?.text().clone();
        let ptr = LocalSyntaxPtr::new(node.syntax());
        Some((name, ptr))
    }
    visitor()
        .visit(decl::<ast::FnDef>)
        .visit(decl::<ast::StructDef>)
        .visit(decl::<ast::EnumDef>)
        .visit(decl::<ast::TraitDef>)
        .visit(decl::<ast::Module>)
        .visit(decl::<ast::TypeDef>)
        .visit(decl::<ast::ConstDef>)
        .visit(decl::<ast::StaticDef>)
        .accept(node)?
}
