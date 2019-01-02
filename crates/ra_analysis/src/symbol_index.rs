use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

use fst::{self, Streamer};
use ra_syntax::{
    AstNode, SyntaxNodeRef, SourceFileNode, SmolStr, TextRange,
    algo::visit::{visitor, Visitor},
    SyntaxKind::{self, *},
    ast::{self, NameOwner, DocCommentsOwner},
};
use ra_db::{SyntaxDatabase, SourceRootId, FilesDatabase};
use salsa::ParallelDatabase;
use rayon::prelude::*;

use crate::{
    Cancelable, FileId, Query,
    db::RootDatabase,
};

salsa::query_group! {
    pub(crate) trait SymbolsDatabase: SyntaxDatabase {
        fn file_symbols(file_id: FileId) -> Cancelable<Arc<SymbolIndex>> {
            type FileSymbolsQuery;
        }
        fn library_symbols(id: SourceRootId) -> Arc<SymbolIndex> {
            type LibrarySymbolsQuery;
            storage input;
        }
    }
}

fn file_symbols(db: &impl SyntaxDatabase, file_id: FileId) -> Cancelable<Arc<SymbolIndex>> {
    db.check_canceled()?;
    let syntax = db.source_file(file_id);
    Ok(Arc::new(SymbolIndex::for_file(file_id, syntax)))
}

pub(crate) fn world_symbols(
    db: &RootDatabase,
    query: Query,
) -> Cancelable<Vec<(FileId, FileSymbol)>> {
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
            .filter_map(|it| it.ok())
            .collect()
    };
    Ok(query.search(&buf))
}

#[derive(Default, Debug)]
pub(crate) struct SymbolIndex {
    symbols: Vec<(FileId, FileSymbol)>,
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
    pub(crate) fn len(&self) -> usize {
        self.symbols.len()
    }

    pub(crate) fn for_files(
        files: impl ParallelIterator<Item = (FileId, SourceFileNode)>,
    ) -> SymbolIndex {
        let mut symbols = files
            .flat_map(|(file_id, file)| {
                file.syntax()
                    .descendants()
                    .filter_map(to_symbol)
                    .map(move |symbol| (symbol.name.as_str().to_lowercase(), (file_id, symbol)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        symbols.par_sort_by(|s1, s2| s1.0.cmp(&s2.0));
        symbols.dedup_by(|s1, s2| s1.0 == s2.0);
        let (names, symbols): (Vec<String>, Vec<(FileId, FileSymbol)>) =
            symbols.into_iter().unzip();
        let map = fst::Map::from_iter(names.into_iter().zip(0u64..)).unwrap();
        SymbolIndex { symbols, map }
    }

    pub(crate) fn for_file(file_id: FileId, file: SourceFileNode) -> SymbolIndex {
        SymbolIndex::for_files(rayon::iter::once((file_id, file)))
    }
}

impl Query {
    pub(crate) fn search(self, indices: &[Arc<SymbolIndex>]) -> Vec<(FileId, FileSymbol)> {
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

                let (file_id, symbol) = &file_symbols.symbols[idx];
                if self.only_types && !is_type(symbol.kind) {
                    continue;
                }
                if self.exact && symbol.name != self.query {
                    continue;
                }
                res.push((*file_id, symbol.clone()));
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct FileSymbol {
    pub(crate) name: SmolStr,
    pub(crate) node_range: TextRange,
    pub(crate) kind: SyntaxKind,
}

impl FileSymbol {
    pub(crate) fn docs(&self, file: &SourceFileNode) -> Option<String> {
        file.syntax()
            .descendants()
            .filter(|node| node.kind() == self.kind && node.range() == self.node_range)
            .filter_map(|node: SyntaxNodeRef| {
                fn doc_comments<'a, N: DocCommentsOwner<'a>>(node: N) -> Option<String> {
                    let comments = node.doc_comment_text();
                    if comments.is_empty() {
                        None
                    } else {
                        Some(comments)
                    }
                }

                visitor()
                    .visit(doc_comments::<ast::FnDef>)
                    .visit(doc_comments::<ast::StructDef>)
                    .visit(doc_comments::<ast::EnumDef>)
                    .visit(doc_comments::<ast::TraitDef>)
                    .visit(doc_comments::<ast::Module>)
                    .visit(doc_comments::<ast::TypeDef>)
                    .visit(doc_comments::<ast::ConstDef>)
                    .visit(doc_comments::<ast::StaticDef>)
                    .accept(node)?
            })
            .nth(0)
    }
    /// Get a description of this node.
    ///
    /// e.g. `struct Name`, `enum Name`, `fn Name`
    pub(crate) fn description(&self, file: &SourceFileNode) -> Option<String> {
        // TODO: After type inference is done, add type information to improve the output
        file.syntax()
            .descendants()
            .filter(|node| node.kind() == self.kind && node.range() == self.node_range)
            .filter_map(|node: SyntaxNodeRef| {
                // TODO: Refactor to be have less repetition
                visitor()
                    .visit(|node: ast::FnDef| {
                        let mut string = "fn ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .visit(|node: ast::StructDef| {
                        let mut string = "struct ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .visit(|node: ast::EnumDef| {
                        let mut string = "enum ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .visit(|node: ast::TraitDef| {
                        let mut string = "trait ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .visit(|node: ast::Module| {
                        let mut string = "mod ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .visit(|node: ast::TypeDef| {
                        let mut string = "type ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .visit(|node: ast::ConstDef| {
                        let mut string = "const ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .visit(|node: ast::StaticDef| {
                        let mut string = "static ".to_string();
                        node.name()?.syntax().text().push_to(&mut string);
                        Some(string)
                    })
                    .accept(node)?
            })
            .nth(0)
    }
}

fn to_symbol(node: SyntaxNodeRef) -> Option<FileSymbol> {
    fn decl<'a, N: NameOwner<'a>>(node: N) -> Option<FileSymbol> {
        let name = node.name()?;
        Some(FileSymbol {
            name: name.text(),
            node_range: node.syntax().range(),
            kind: node.syntax().kind(),
        })
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
