use std::{
    hash::Hash,
    sync::Arc,
    cell::RefCell,
    fmt::Debug,
};
use parking_lot::Mutex;
use libsyntax2::{File};
use im;
use {
    FileId,
    imp::{FileResolverImp},
    module_map_db::ModuleDescr,
};

#[derive(Debug)]
pub(crate) struct DbHost {
    db: Arc<Db>,
}

impl DbHost {
    pub(crate) fn new() -> DbHost {
        let db = Db {
            file_resolver: FileResolverImp::default(),
            files: im::HashMap::new(),
            cache: Mutex::new(Cache::new())
        };
        DbHost { db: Arc::new(db) }
    }
    pub(crate) fn change_file(&mut self, file_id: FileId, text: Option<String>) {
        let db = self.db_mut();
        match text {
            None => {
                db.files.remove(&file_id);
            }
            Some(text) => {
                db.files.insert(file_id, Arc::new(text));
            }
        }
    }
    pub(crate) fn set_file_resolver(&mut self, file_resolver: FileResolverImp) {
        let db = self.db_mut();
        db.file_resolver = file_resolver
    }
    pub(crate) fn query_ctx(&self) -> QueryCtx {
        QueryCtx {
            db: Arc::clone(&self.db),
            trace: RefCell::new(Vec::new()),
        }
    }
    fn db_mut(&mut self) -> &mut Db {
        // NB: this "forks" the database & clears the cache
        let db = Arc::make_mut(&mut self.db);
        *db.cache.get_mut() = Default::default();
        db
    }
}

#[derive(Debug)]
pub(crate) struct Db {
    file_resolver: FileResolverImp,
    files: im::HashMap<FileId, Arc<String>>,
    cache: Mutex<Cache>,
}

impl Clone for Db {
    fn clone(&self) -> Db {
        Db {
            file_resolver: self.file_resolver.clone(),
            files: self.files.clone(),
            cache: Mutex::new(Cache::new()),
        }
    }
}

#[derive(Clone, Default, Debug)]
pub(crate) struct Cache {
    pub(crate) module_descr: QueryCache<ModuleDescr>
}
#[allow(type_alias_bounds)]
pub(crate) type QueryCache<Q: Query> = im::HashMap<
    <Q as Query>::Params,
    <Q as Query>::Output
>;

impl Cache {
    fn new() -> Cache {
        Default::default()
    }
}

pub(crate) struct QueryCtx {
    db: Arc<Db>,
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
        let res = Q::get(self, params);
        res
    }
    fn trace(&self, event: TraceEvent) {
        self.trace.borrow_mut().push(event)
    }
}

pub(crate) trait Query {
    const ID: u32;
    type Params: Hash + Eq + Debug;
    type Output: Debug;
}

pub(crate) trait Get: Query {
    fn get(ctx: &QueryCtx, params: &Self::Params) -> Self::Output;
}

impl<T: Eval> Get for T
where
    T::Params: Clone,
    T::Output: Clone,
{
    fn get(ctx: &QueryCtx, params: &Self::Params) -> Self::Output {
        {
            let mut cache = ctx.db.cache.lock();
            if let Some(cache) = Self::cache(&mut cache) {
                if let Some(res) = cache.get(params) {
                    return res.clone();
                }
            }
        }
        ctx.trace(TraceEvent { query_id: Self::ID, kind: TraceEventKind::Start });
        let res = Self::eval(ctx, params);
        ctx.trace(TraceEvent { query_id: Self::ID, kind: TraceEventKind::Finish });

        let mut cache = ctx.db.cache.lock();
        if let Some(cache) = Self::cache(&mut cache) {
            cache.insert(params.clone(), res.clone());
        }

        res
    }
}

pub(crate) trait Eval: Query
where
    Self::Params: Clone,
    Self::Output: Clone,
 {
    fn cache(_cache: &mut Cache) -> Option<&mut QueryCache<Self>> {
        None
    }
    fn eval(ctx: &QueryCtx, params: &Self::Params) -> Self::Output;
}

#[derive(Debug)]
pub(crate) struct DbFiles {
    db: Arc<Db>,
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
        DbFiles { db: Arc::clone(&ctx.db) }
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
