//! This module handles fuzzy-searching of functions, structs and other symbols
//! by name across the whole workspace and dependencies.
//!
//! It works by building an incrementally-updated text-search index of all
//! symbols. The backbone of the index is the **awesome** `fst` crate by
//! @BurntSushi.
//!
//! In a nutshell, you give a set of strings to `fst`, and it builds a
//! finite state machine describing this set of strings. The strings which
//! could fuzzy-match a pattern can also be described by a finite state machine.
//! What is freaking cool is that you can now traverse both state machines in
//! lock-step to enumerate the strings which are both in the input set and
//! fuzz-match the query. Or, more formally, given two languages described by
//! FSTs, one can build a product FST which describes the intersection of the
//! languages.
//!
//! `fst` does not support cheap updating of the index, but it supports unioning
//! of state machines. So, to account for changing source code, we build an FST
//! for each library (which is assumed to never change) and an FST for each Rust
//! file in the current workspace, and run a query against the union of all
//! those FSTs.
use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
    sync::Arc,
    mem,
    fmt,
};

use fst::{self, Streamer};
use ra_syntax::{
    SyntaxNode, SyntaxNodePtr, SourceFile, SmolStr, TreeArc, AstNode,
    algo::{visit::{visitor, Visitor}},
    SyntaxKind::{self, *},
    ast::{self, NameOwner},
    WalkEvent,
    TextRange,
};
use ra_db::{
    SourceRootId, SourceDatabase,
    salsa::{self, ParallelDatabase},
};
use rayon::prelude::*;

use crate::{
    FileId, Query,
    db::RootDatabase,
};

#[salsa::query_group(SymbolsDatabaseStorage)]
pub(crate) trait SymbolsDatabase: hir::db::HirDatabase {
    fn file_symbols(&self, file_id: FileId) -> Arc<SymbolIndex>;
    #[salsa::input]
    fn library_symbols(&self, id: SourceRootId) -> Arc<SymbolIndex>;
    /// The set of "local" (that is, from the current workspace) roots.
    /// Files in local roots are assumed to change frequently.
    #[salsa::input]
    fn local_roots(&self) -> Arc<Vec<SourceRootId>>;
    /// The set of roots for crates.io libraries.
    /// Files in libraries are assumed to never change.
    #[salsa::input]
    fn library_roots(&self) -> Arc<Vec<SourceRootId>>;
}

fn file_symbols(db: &impl SymbolsDatabase, file_id: FileId) -> Arc<SymbolIndex> {
    db.check_canceled();
    let source_file = db.parse(file_id);

    let symbols = source_file_to_file_symbols(&source_file, file_id);

    // TODO: add macros here

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
        files.par_iter().map_with(snap, |db, &file_id| db.0.file_symbols(file_id)).collect()
    };
    query.search(&buf)
}

pub(crate) fn index_resolve(db: &RootDatabase, name_ref: &ast::NameRef) -> Vec<FileSymbol> {
    let name = name_ref.text();
    let mut query = Query::new(name.to_string());
    query.exact();
    query.limit(4);
    crate::symbol_index::world_symbols(db, query)
}

#[derive(Default)]
pub(crate) struct SymbolIndex {
    symbols: Vec<FileSymbol>,
    map: fst::Map,
}

impl fmt::Debug for SymbolIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SymbolIndex").field("n_symbols", &self.symbols.len()).finish()
    }
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
        let map = fst::Map::from_iter(names.zip(0u64..)).unwrap();
        SymbolIndex { symbols, map }
    }

    pub(crate) fn len(&self) -> usize {
        self.symbols.len()
    }

    pub(crate) fn memory_size(&self) -> usize {
        self.map.as_fst().size() + self.symbols.len() * mem::size_of::<FileSymbol>()
    }

    pub(crate) fn for_files(
        files: impl ParallelIterator<Item = (FileId, TreeArc<SourceFile>)>,
    ) -> SymbolIndex {
        let symbols = files
            .flat_map(|(file_id, file)| source_file_to_file_symbols(&file, file_id))
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
        STRUCT_DEF | ENUM_DEF | TRAIT_DEF | TYPE_ALIAS_DEF => true,
        _ => false,
    }
}

/// The actual data that is stored in the index. It should be as compact as
/// possible.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FileSymbol {
    pub(crate) file_id: FileId,
    pub(crate) name: SmolStr,
    pub(crate) ptr: SyntaxNodePtr,
    pub(crate) name_range: Option<TextRange>,
    pub(crate) container_name: Option<SmolStr>,
}

fn source_file_to_file_symbols(source_file: &SourceFile, file_id: FileId) -> Vec<FileSymbol> {
    let mut symbols = Vec::new();
    let mut stack = Vec::new();

    for event in source_file.syntax().preorder() {
        match event {
            WalkEvent::Enter(node) => {
                if let Some(mut symbol) = to_file_symbol(node, file_id) {
                    symbol.container_name = stack.last().cloned();

                    stack.push(symbol.name.clone());
                    symbols.push(symbol);
                }
            }

            WalkEvent::Leave(node) => {
                if to_symbol(node).is_some() {
                    stack.pop();
                }
            }
        }
    }

    symbols
}

fn to_symbol(node: &SyntaxNode) -> Option<(SmolStr, SyntaxNodePtr, TextRange)> {
    fn decl<N: NameOwner>(node: &N) -> Option<(SmolStr, SyntaxNodePtr, TextRange)> {
        let name = node.name()?;
        let name_range = name.syntax().range();
        let name = name.text().clone();
        let ptr = SyntaxNodePtr::new(node.syntax());

        Some((name, ptr, name_range))
    }
    visitor()
        .visit(decl::<ast::FnDef>)
        .visit(decl::<ast::StructDef>)
        .visit(decl::<ast::EnumDef>)
        .visit(decl::<ast::TraitDef>)
        .visit(decl::<ast::Module>)
        .visit(decl::<ast::TypeAliasDef>)
        .visit(decl::<ast::ConstDef>)
        .visit(decl::<ast::StaticDef>)
        .accept(node)?
}

fn to_file_symbol(node: &SyntaxNode, file_id: FileId) -> Option<FileSymbol> {
    to_symbol(node).map(move |(name, ptr, name_range)| FileSymbol {
        name,
        ptr,
        file_id,
        name_range: Some(name_range),
        container_name: None,
    })
}
