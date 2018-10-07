use std::{
    fmt,
    sync::Arc,
    hash::{Hash, Hasher},
    collections::HashSet,
};
use salsa;
use ra_syntax::File;
use ra_editor::{LineIndex};
use crate::{
    symbol_index::SymbolIndex,
    FileId, FileResolverImp
};

#[derive(Default)]
pub(crate) struct RootDatabase {
    runtime: salsa::runtime::Runtime<RootDatabase>,
}

impl fmt::Debug for RootDatabase {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.write_str("RootDatabase { ... }")
    }
}

impl salsa::Database for RootDatabase {
    fn salsa_runtime(&self) -> &salsa::runtime::Runtime<RootDatabase> {
        &self.runtime
    }
}

salsa::database_storage! {
    pub(crate) struct RootDatabaseStorage for RootDatabase {
        impl FilesDatabase {
            fn file_text() for FileTextQuery;
            fn file_set() for FileSetQuery;
        }
        impl SyntaxDatabase {
            fn file_syntax() for FileSyntaxQuery;
            fn file_lines() for FileLinesQuery;
            fn file_symbols() for FileSymbolsQuery;
        }
    }
}

salsa::query_group! {
    pub(crate) trait FilesDatabase: salsa::Database {
        fn file_text(file_id: FileId) -> Arc<String> {
            type FileTextQuery;
            storage input;
        }
        fn file_set(key: ()) -> Arc<FileSet> {
            type FileSetQuery;
            storage input;
        }
    }
}

#[derive(Default, Debug)]
pub(crate) struct FileSet {
    pub(crate) files: HashSet<FileId>,
    pub(crate) resolver: FileResolverImp,
}

impl PartialEq for FileSet {
    fn eq(&self, other: &FileSet) -> bool {
        self.files == other.files
    }
}

impl Eq for FileSet {
}

impl Hash for FileSet {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        let mut files = self.files.iter().cloned().collect::<Vec<_>>();
        files.sort();
        files.hash(hasher);
    }
}

salsa::query_group! {
    pub(crate) trait SyntaxDatabase: FilesDatabase {
        fn file_syntax(file_id: FileId) -> File {
            type FileSyntaxQuery;
        }
        fn file_lines(file_id: FileId) -> Arc<LineIndex> {
            type FileLinesQuery;
        }
        fn file_symbols(file_id: FileId) -> Arc<SymbolIndex> {
            type FileSymbolsQuery;
        }
    }
}

fn file_syntax(db: &impl SyntaxDatabase, file_id: FileId) -> File {
    let text = db.file_text(file_id);
    File::parse(&*text)
}
fn file_lines(db: &impl SyntaxDatabase, file_id: FileId) -> Arc<LineIndex> {
    let text = db.file_text(file_id);
    Arc::new(LineIndex::new(&*text))
}
fn file_symbols(db: &impl SyntaxDatabase, file_id: FileId) -> Arc<SymbolIndex> {
    let syntax = db.file_syntax(file_id);
    Arc::new(SymbolIndex::for_file(file_id, syntax))
}

// mod imp;

// use std::{
//     sync::Arc,
// };
// use im;
// use salsa;
// use {FileId, imp::FileResolverImp};

// #[derive(Debug, Default, Clone)]
// pub(crate) struct State {
//     pub(crate) file_map: im::HashMap<FileId, Arc<String>>,
//     pub(crate) file_resolver: FileResolverImp
// }

// #[derive(Debug)]
// pub(crate) struct Db {
//     imp: imp::Db,
// }

// #[derive(Clone, Copy)]
// pub(crate) struct QueryCtx<'a> {
//     imp: &'a salsa::QueryCtx<State, imp::Data>,
// }

// pub(crate) struct Query<T, R>(pub(crate) u16, pub(crate) fn(QueryCtx, &T) -> R);

// pub(crate) struct QueryRegistry {
//     imp: imp::QueryRegistry,
// }

// impl Default for Db {
//     fn default() -> Db {
//         Db::new()
//     }
// }

// impl Db {
//     pub(crate) fn new() -> Db {
//         let reg = QueryRegistry::new();
//         Db { imp: imp::Db::new(reg.imp) }
//     }
//     pub(crate) fn state(&self) -> &State {
//         self.imp.imp.ground_data()
//     }
//     pub(crate) fn with_changes(&self, new_state: State, changed_files: &[FileId], resolver_changed: bool) -> Db {
//         Db { imp: self.imp.with_changes(new_state, changed_files, resolver_changed) }
//     }
//     pub(crate) fn make_query<F: FnOnce(QueryCtx) -> R, R>(&self, f: F) -> R {
//         let ctx = QueryCtx { imp: &self.imp.imp.query_ctx() };
//         f(ctx)
//     }
//     #[allow(unused)]
//     pub(crate) fn trace_query<F: FnOnce(QueryCtx) -> R, R>(&self, f: F) -> (R, Vec<&'static str>) {
//         let ctx = QueryCtx { imp: &self.imp.imp.query_ctx() };
//         let res = f(ctx);
//         let trace = self.imp.extract_trace(ctx.imp);
//         (res, trace)
//     }
// }

// impl<'a> QueryCtx<'a> {
//     pub(crate) fn get<Q: imp::EvalQuery>(&self, q: Q, params: Q::Params) -> Arc<Q::Output> {
//         q.get(self, params)
//     }
// }

// pub(crate) fn file_text(ctx: QueryCtx, file_id: FileId) -> Arc<String> {
//     imp::file_text(ctx, file_id)
// }

// pub(crate) fn file_set(ctx: QueryCtx) -> Arc<(Vec<FileId>, FileResolverImp)> {
//     imp::file_set(ctx)
// }
// impl QueryRegistry {
//     fn new() -> QueryRegistry {
//         let mut reg = QueryRegistry { imp: imp::QueryRegistry::new() };
//         ::queries::register_queries(&mut reg);
//         ::module_map::register_queries(&mut reg);
//         reg
//     }
//     pub(crate) fn add<Q: imp::EvalQuery>(&mut self, q: Q, name: &'static str) {
//         self.imp.add(q, name)
//     }
// }
