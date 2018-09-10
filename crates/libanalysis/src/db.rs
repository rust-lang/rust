use std::{
    hash::Hash,
    sync::Arc,
    cell::RefCell,
};
use libsyntax2::{File};
use im;
use {
    FileId,
    imp::{FileResolverImp},
};

#[derive(Clone)]
pub(crate) struct Db {
    file_resolver: FileResolverImp,
    files: im::HashMap<FileId, Arc<String>>,
}

impl Db {
    pub(crate) fn new() -> Db {
        Db {
            file_resolver: FileResolverImp::default(),
            files: im::HashMap::new(),
        }
    }
    pub(crate) fn change_file(&mut self, file_id: FileId, text: Option<String>) {
        match text {
            None => {
                self.files.remove(&file_id);
            }
            Some(text) => {
                self.files.insert(file_id, Arc::new(text));
            }
        }
    }
    pub(crate) fn set_file_resolver(&mut self, file_resolver: FileResolverImp) {
        self.file_resolver = file_resolver
    }
    pub(crate) fn query_ctx(&self) -> QueryCtx {
        QueryCtx {
            db: self.clone(),
            trace: RefCell::new(Vec::new()),
        }
    }
}

pub(crate) struct QueryCtx {
    db: Db,
    pub(crate) trace: RefCell<Vec<TraceEvent>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TraceEvent {
    pub(crate) query_id: u32,
    pub(crate) kind: TraceEventKind
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TraceEventKind {
    Start, Finish
}

impl QueryCtx {
    pub(crate) fn get<Q: Get>(&self, params: &Q::Params) -> Q::Output {
        self.trace(TraceEvent { query_id: Q::ID, kind: TraceEventKind::Start });
        let res = Q::get(self, params);
        self.trace(TraceEvent { query_id: Q::ID, kind: TraceEventKind::Finish });
        res
    }
    fn trace(&self, event: TraceEvent) {
        self.trace.borrow_mut().push(event)
    }
}

pub(crate) trait Query {
    const ID: u32;
    type Params: Hash;
    type Output;
}

pub(crate) trait Get: Query {
    fn get(ctx: &QueryCtx, params: &Self::Params) -> Self::Output;
}

impl<T: Eval> Get for T {
    fn get(ctx: &QueryCtx, params: &Self::Params) -> Self::Output {
        Self::eval(ctx, params)
    }
}

pub(crate) trait Eval: Query {
    fn eval(ctx: &QueryCtx, params: &Self::Params) -> Self::Output;
}

pub(crate) struct DbFiles {
    db: Db,
}

impl DbFiles {
    pub(crate) fn iter<'a>(&'a self) -> impl Iterator<Item=FileId> + 'a {
        self.db.files.keys().cloned()
    }
    pub(crate) fn file_resolver(&self) -> FileResolverImp {
        self.db.file_resolver.clone()
    }
}

pub(crate) enum Files {}
impl Query for Files {
    const ID: u32 = 1;
    type Params = ();
    type Output = DbFiles;
}
impl Get for Files {
    fn get(ctx: &QueryCtx, _params: &()) -> DbFiles {
        DbFiles { db: ctx.db.clone() }
    }
}

enum FileText {}
impl Query for FileText {
    const ID: u32 = 10;
    type Params = FileId;
    type Output = Arc<String>;
}
impl Get for FileText {
    fn get(ctx: &QueryCtx, file_id: &FileId) -> Arc<String> {
        ctx.db.files[file_id].clone()
    }
}

pub(crate) enum FileSyntax {}
impl Query for FileSyntax {
    const ID: u32 = 20;
    type Params = FileId;
    type Output = File;
}
impl Eval for FileSyntax {
    fn eval(ctx: &QueryCtx, file_id: &FileId) -> File {
        let text = ctx.get::<FileText>(file_id);
        File::parse(&text)
    }
}
