use std::{
    sync::Arc,
    any::Any,
    hash::{Hash, Hasher},
    collections::hash_map::{DefaultHasher, HashMap},
    iter,
};
use salsa;
use {FileId, imp::FileResolverImp};
use super::{State, Query, QueryCtx};

pub(super) type Data = Arc<Any + Send + Sync + 'static>;

#[derive(Debug)]
pub(super) struct Db {
    names: Arc<HashMap<salsa::QueryTypeId, &'static str>>,
    pub(super) imp: salsa::Db<State, Data>,
}

impl Db {
    pub(super) fn new(mut reg: QueryRegistry) -> Db {
        let config = reg.config.take().unwrap();
        Db {
            names: Arc::new(reg.names),
            imp: salsa::Db::new(config, State::default())
        }
    }
    pub(crate) fn with_changes(&self, new_state: State, changed_files: &[FileId], resolver_changed: bool) -> Db {
        let names = self.names.clone();
        let mut invalidations = salsa::Invalidations::new();
        invalidations.invalidate(FILE_TEXT, changed_files.iter().map(hash).map(salsa::InputFingerprint));
        if resolver_changed {
            invalidations.invalidate(FILE_SET, iter::once(salsa::InputFingerprint(hash(&()))));
        } else {
            invalidations.invalidate(FILE_SET, iter::empty());
        }
        let imp = self.imp.with_ground_data(
            new_state,
            invalidations,
        );
        Db { names, imp }
    }
    pub(super) fn extract_trace(&self, ctx: &salsa::QueryCtx<State, Data>) -> Vec<&'static str> {
        ctx.trace().into_iter().map(|it| self.names[&it]).collect()
    }
}

pub(crate) trait EvalQuery {
    type Params;
    type Output;
    fn query_type(&self) -> salsa::QueryTypeId;
    fn f(&self) -> salsa::QueryFn<State, Data>;
    fn get(&self, &QueryCtx, Self::Params) -> Arc<Self::Output>;
}

impl<T, R> EvalQuery for Query<T, R>
where
    T: Hash + Send + Sync + 'static,
    R: Hash + Send + Sync + 'static,
{
    type Params = T;
    type Output = R;
    fn query_type(&self) -> salsa::QueryTypeId {
        salsa::QueryTypeId(self.0)
    }
    fn f(&self) -> salsa::QueryFn<State, Data> {
        let f = self.1;
        Box::new(move |ctx, data| {
            let ctx = QueryCtx { imp: ctx };
            let data: &T = data.downcast_ref().unwrap();
            let res = f(ctx, data);
            let h = hash(&res);
            (Arc::new(res), salsa::OutputFingerprint(h))
        })
    }
    fn get(&self, ctx: &QueryCtx, params: Self::Params) -> Arc<Self::Output> {
        let query_id = salsa::QueryId(
            self.query_type(),
            salsa::InputFingerprint(hash(&params)),
        );
        let res = ctx.imp.get(query_id, Arc::new(params));
        res.downcast().unwrap()
    }
}

pub(super) struct QueryRegistry {
    config: Option<salsa::QueryConfig<State, Data>>,
    names: HashMap<salsa::QueryTypeId, &'static str>,
}

impl QueryRegistry {
    pub(super) fn new() -> QueryRegistry {
        let mut config = salsa::QueryConfig::<State, Data>::new();
        config = config.with_ground_query(
            FILE_TEXT, Box::new(|state, params| {
                let file_id: &FileId = params.downcast_ref().unwrap();
                let res = state.file_map[file_id].clone();
                let fingerprint = salsa::OutputFingerprint(hash(&res));
                (res, fingerprint)
            })
        );
        config = config.with_ground_query(
            FILE_SET, Box::new(|state, _params| {
                let file_ids: Vec<FileId> = state.file_map.keys().cloned().collect();
                let hash = hash(&file_ids);
                let file_resolver = state.file_resolver.clone();
                let res = (file_ids, file_resolver);
                let fingerprint = salsa::OutputFingerprint(hash);
                (Arc::new(res), fingerprint)
            })
        );
        let mut names = HashMap::new();
        names.insert(FILE_TEXT, "FILE_TEXT");
        names.insert(FILE_SET, "FILE_SET");
        QueryRegistry { config: Some(config), names }
    }
    pub(super) fn add<Q: EvalQuery>(&mut self, q: Q, name: &'static str) {
        let id = q.query_type();
        let prev = self.names.insert(id, name);
        assert!(prev.is_none(), "duplicate query: {:?}", id);
        let config = self.config.take().unwrap();
        let config = config.with_query(id, q.f());
        self.config= Some(config);
    }
    pub(super) fn finish(mut self) -> salsa::QueryConfig<State, Data> {
        self.config.take().unwrap()
    }
}

fn hash<T: Hash>(x: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    hasher.finish()
}

const FILE_TEXT: salsa::QueryTypeId = salsa::QueryTypeId(0);
pub(super) fn file_text(ctx: QueryCtx, file_id: FileId) -> Arc<String> {
    let query_id = salsa::QueryId(
        FILE_TEXT,
        salsa::InputFingerprint(hash(&file_id)),
    );
    let res = ctx.imp.get(query_id, Arc::new(file_id));
    res.downcast().unwrap()
}

const FILE_SET: salsa::QueryTypeId = salsa::QueryTypeId(1);
pub(super) fn file_set(ctx: QueryCtx) -> Arc<(Vec<FileId>, FileResolverImp)> {
    let query_id = salsa::QueryId(
        FILE_SET,
        salsa::InputFingerprint(hash(&())),
    );
    let res = ctx.imp.get(query_id, Arc::new(()));
    res.downcast().unwrap()
}

