use std::{
    hash::{Hash, Hasher},
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
            stack: RefCell::new(Vec::new()),
            trace: RefCell::new(Vec::new()),
        }
    }
    fn db_mut(&mut self) -> &mut Db {
        // NB: this "forks" the database
        let db = Arc::make_mut(&mut self.db);
        db.cache.get_mut().gen += 1;
        db
    }
}

type QueryInvocationId = (u32, u64);
type Gen = u64;
type OutputHash = u64;

fn id<Q: Query>(params: &Q::Params) -> QueryInvocationId {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    params.hash(&mut hasher);
    (Q::ID, hasher.finish())
}
fn output_hash<Q: Query>(output: &Q::Output) -> OutputHash {
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    output.hash(&mut hasher);
    hasher.finish()
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


#[derive(Default, Debug)]
pub(crate) struct Cache {
    pub(crate) module_descr: QueryCache<ModuleDescr>,
    gen: Gen,
    green: im::HashMap<QueryInvocationId, (Gen, OutputHash)>,
    deps: im::HashMap<QueryInvocationId, Vec<(QueryInvocationId, OutputHash)>>,
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
    stack: RefCell<Vec<QueryInvocationId>>,
    pub(crate) trace: RefCell<Vec<TraceEvent>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct TraceEvent {
    pub(crate) query_id: u32,
    pub(crate) kind: TraceEventKind
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TraceEventKind {
    Start, Evaluating, Finish
}

impl QueryCtx {
    pub(crate) fn get<Q: Get>(&self, params: &Q::Params) -> Q::Output {
        let me = id::<Q>(params);
        eprintln!("eval: {:?}", me);
        let parent = self.stack.borrow().last().map(|&id| id);
        self.stack.borrow_mut().push(me);
        self.trace(TraceEvent { query_id: Q::ID, kind: TraceEventKind::Start });
        let res = Q::get(self, params);
        self.trace(TraceEvent { query_id: Q::ID, kind: TraceEventKind::Finish });
        if let Some(parent) = parent {
            let h = output_hash::<Q>(&res);
            let mut cache = self.db.cache.lock();
            cache.deps
                .entry(parent)
                .or_insert(Vec::new())
                .push((me, h))
        }
        let also_me = self.stack.borrow_mut().pop();
        assert_eq!(also_me, Some(me));
        res
    }
    fn trace(&self, event: TraceEvent) {
        self.trace.borrow_mut().push(event)
    }
}

pub(crate) trait Query {
    const ID: u32;
    type Params: Hash + Eq + Debug;
    type Output: Hash + Debug;
}

pub(crate) trait Get: Query {
    fn get(ctx: &QueryCtx, params: &Self::Params) -> Self::Output;
}

impl<Q: Eval> Get for Q
where
    Q::Params: Clone,
    Q::Output: Clone,
{
    fn get(ctx: &QueryCtx, params: &Self::Params) -> Self::Output {
        if !Self::cacheable() {
            ctx.trace(TraceEvent { query_id: Q::ID, kind: TraceEventKind::Evaluating });
            return Self::eval(ctx, params);
        }

        if let Some(res) = try_reuse::<Q>(ctx, params) {
            return res;
        }

        ctx.trace(TraceEvent { query_id: Q::ID, kind: TraceEventKind::Evaluating });
        let res = Self::eval(ctx, params);

        let mut cache = ctx.db.cache.lock();
        let gen = cache.gen;
        let output_hash = output_hash::<Q>(&res);
        let id = id::<Q>(params);
        cache.green.insert(id, (gen, output_hash));
        let cache = Self::cache(&mut cache);
        cache.insert(params.clone(), res.clone());
        res
    }
}

fn try_reuse<Q: Eval>(ctx: &QueryCtx, params: &Q::Params) -> Option<Q::Output>
where
    Q::Params: Clone,
    Q::Output: Clone,
{
    let id = id::<Q>(params);
    let mut cache = ctx.db.cache.lock();
    let curr_gen = cache.gen;
    let old_hash = match *cache.green.get(&id)? {
        (gen, _) if gen == curr_gen => {
            return Some(Q::cache(&mut cache)[params].clone());
        }
        (_, hash) => hash,
    };
    let deps_are_fresh = cache.deps[&id]
        .iter()
        .all(|&(dep_id, dep_hash)| {
            match cache.green.get(&dep_id) {
                //TODO: store the value of parameters, and re-execute the query
                Some((gen, hash)) if gen == &curr_gen && hash == &dep_hash => true,
                _ => false,
            }
        });
    if !deps_are_fresh {
        return None;
    }
    cache.green.insert(id, (curr_gen, old_hash));
    Some(Q::cache(&mut cache)[params].clone())
}

pub(crate) trait Eval: Query
where
    Self::Params: Clone,
    Self::Output: Clone,
{
    fn cacheable() -> bool { false }
    fn cache(_cache: &mut Cache) -> &mut QueryCache<Self> {
        unimplemented!()
    }
    fn eval(ctx: &QueryCtx, params: &Self::Params) -> Self::Output;
}

#[derive(Debug)]
pub(crate) struct DbFiles {
    db: Arc<Db>,
}

impl Hash for DbFiles {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.db.cache.lock().gen.hash(hasher)
    }
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
    fn get(ctx: &QueryCtx, params: &()) -> DbFiles {
        let res = DbFiles { db: Arc::clone(&ctx.db) };
        let id = id::<Self>(params);
        let hash = output_hash::<Self>(&res);
        let mut cache = ctx.db.cache.lock();
        let gen = cache.gen;
        cache.green.insert(id, (gen, hash));
        res
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
        let res = ctx.db.files[file_id].clone();
        let id = id::<Self>(file_id);
        let hash = output_hash::<Self>(&res);
        let mut cache = ctx.db.cache.lock();
        let gen = cache.gen;
        cache.green.insert(id, (gen, hash));
        res
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
