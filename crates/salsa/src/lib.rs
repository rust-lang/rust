extern crate im;
extern crate parking_lot;

use std::{
    sync::Arc,
    collections::{HashSet, HashMap},
    cell::RefCell,
};
use parking_lot::Mutex;

type GroundQueryFn<T, D> = Box<Fn(&T, &D) -> (D, OutputFingerprint) + Send + Sync + 'static>;
type QueryFn<T, D> = Box<Fn(&QueryCtx<T, D>, &D) -> (D, OutputFingerprint) + Send + Sync + 'static>;

#[derive(Debug)]
pub struct Db<T, D> {
    db: Arc<DbState<T, D>>,
    query_config: Arc<QueryConfig<T, D>>,
}

pub struct QueryConfig<T, D> {
    ground_fn: HashMap<QueryTypeId, GroundQueryFn<T, D>>,
    query_fn: HashMap<QueryTypeId, QueryFn<T, D>>,
}

impl<T, D> ::std::fmt::Debug for QueryConfig<T, D> {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        ::std::fmt::Display::fmt("QueryConfig { ... }", f)
    }
}

#[derive(Debug)]
struct DbState<T, D> {
    ground_data: T,
    gen: Gen,
    graph: Mutex<im::HashMap<QueryId, (Gen, Arc<QueryRecord<D>>)>>,
}

#[derive(Debug)]
struct QueryRecord<D> {
    params: D,
    output: D,
    output_fingerprint: OutputFingerprint,
    deps: Vec<(QueryId, OutputFingerprint)>,
}

impl<T, D> DbState<T, D> {
    fn record(
        &self,
        query_id: QueryId,
        params: D,
        output: D,
        output_fingerprint: OutputFingerprint,
        deps: Vec<(QueryId, OutputFingerprint)>,
    ) {
        let gen = self.gen;
        let record = QueryRecord {
            params,
            output,
            output_fingerprint,
            deps,
        };
        self.graph.lock().insert(query_id, (gen, Arc::new(record)));
    }
}

impl<T, D> QueryConfig<T, D> {
    pub fn new() -> Self {
        QueryConfig {
            ground_fn: HashMap::new(),
            query_fn: HashMap::new(),
        }
    }
    pub fn with_ground_query(
        mut self,
        query_type: QueryTypeId,
        query_fn: GroundQueryFn<T, D>
    ) -> Self {
        let prev = self.ground_fn.insert(query_type, query_fn);
        assert!(prev.is_none());
        self
    }
    pub fn with_query(
        mut self,
        query_type: QueryTypeId,
        query_fn: QueryFn<T, D>,
    ) -> Self {
        let prev = self.query_fn.insert(query_type, query_fn);
        assert!(prev.is_none());
        self
    }
}

pub struct QueryCtx<T, D> {
    db: Arc<DbState<T, D>>,
    query_config: Arc<QueryConfig<T, D>>,
    stack: RefCell<Vec<Vec<(QueryId, OutputFingerprint)>>>,
    executed: RefCell<Vec<QueryTypeId>>,
}

impl<T, D> QueryCtx<T, D>
where
    D: Clone
{
    fn new(db: &Db<T, D>) -> QueryCtx<T, D> {
        QueryCtx {
            db: Arc::clone(&db.db),
            query_config: Arc::clone(&db.query_config),
            stack: RefCell::new(vec![Vec::new()]),
            executed: RefCell::new(Vec::new()),
        }
    }
    pub fn get(
        &self,
        query_id: QueryId,
        params: D,
    ) -> D {
        let (res, output_fingerprint) = self.get_inner(query_id, params);
        self.record_dep(query_id, output_fingerprint);
        res
    }

    fn get_inner(
        &self,
        query_id: QueryId,
        params: D,
    ) -> (D, OutputFingerprint) {
        let (gen, record) = {
            let guard = self.db.graph.lock();
            match guard.get(&query_id).map(|it| it.clone()){
                None => {
                    drop(guard);
                    return self.force(query_id, params);
                },
                Some(it) => it,
            }
        };
        if gen == self.db.gen {
            return (record.output.clone(), record.output_fingerprint)
        }
        if self.query_config.ground_fn.contains_key(&query_id.0) {
            let (invalidated, record) = {
                let guard = self.db.graph.lock();
                let (gen, ref record) = guard[&query_id];
                (gen == INVALIDATED, record.clone())
            };
            if invalidated {
                return self.force(query_id, params);
            } else {
                return (record.output.clone(), record.output_fingerprint);
            }
        }
        for (dep_query_id, prev_fingerprint) in record.deps.iter().cloned() {
            let dep_params: D = {
                let guard = self.db.graph.lock();
                guard[&dep_query_id]
                .1
                .params
                .clone()
            };
            if prev_fingerprint != self.get_inner(dep_query_id, dep_params).1 {
                return self.force(query_id, params)
            }
        }
        let gen = self.db.gen;
        {
            let mut guard = self.db.graph.lock();
            guard[&query_id].0 = gen;
        }
        (record.output.clone(), record.output_fingerprint)
    }
    fn force(
        &self,
        query_id: QueryId,
        params: D,
    ) -> (D, OutputFingerprint) {
        self.executed.borrow_mut().push(query_id.0);
        self.stack.borrow_mut().push(Vec::new());

        let (res, output_fingerprint) = if let Some(f) = self.query_config.ground_fn.get(&query_id.0) {
            f(&self.db.ground_data, &params)
        } else if let Some(f) = self.query_config.query_fn.get(&query_id.0) {
            f(self, &params)
        } else {
            panic!("unknown query type: {:?}", query_id.0);
        };

        let res: D = res.into();

        let deps = self.stack.borrow_mut().pop().unwrap();
        self.db.record(query_id, params, res.clone(), output_fingerprint, deps);
        (res, output_fingerprint)
    }
    fn record_dep(
        &self,
        query_id: QueryId,
        output_fingerprint: OutputFingerprint,
    ) -> () {
        let mut stack = self.stack.borrow_mut();
        let deps = stack.last_mut().unwrap();
        deps.push((query_id, output_fingerprint))
    }
}

pub struct Invalidations {
    types: HashSet<QueryTypeId>,
    ids: Vec<QueryId>,
}

impl Invalidations {
    pub fn new() -> Invalidations {
        Invalidations {
            types: HashSet::new(),
            ids: Vec::new(),
        }
    }
    pub fn invalidate(
        &mut self,
        query_type: QueryTypeId,
        params: impl Iterator<Item=InputFingerprint>,
    ) {
        self.types.insert(query_type);
        self.ids.extend(params.map(|it| QueryId(query_type, it)))
    }
}

impl<T, D> Db<T, D>
where
    D: Clone
{
    pub fn new(query_config: QueryConfig<T, D>, ground_data: T) -> Db<T, D> {
        Db {
            db: Arc::new(DbState { ground_data, gen: Gen(0), graph: Default::default() }),
            query_config: Arc::new(query_config),
        }
    }
    pub fn ground_data(&self) -> &T {
        &self.db.ground_data
    }
    pub fn with_ground_data(
        &self,
        ground_data: T,
        invalidations: Invalidations,
    ) -> Db<T, D> {
        for id in self.query_config.ground_fn.keys() {
            assert!(
                invalidations.types.contains(id),
                "all ground queries must be invalidated"
            );
        }

        let gen = Gen(self.db.gen.0 + 1);
        let mut graph = self.db.graph.lock().clone();
        for id in invalidations.ids {
            if let Some((gen, _)) = graph.get_mut(&id) {
                *gen = INVALIDATED;
            }
        }
        let graph = Mutex::new(graph);
        Db {
            db: Arc::new(DbState { ground_data, gen, graph }),
            query_config: Arc::clone(&self.query_config)
        }
    }
    pub fn get(
        &self,
        query_id: QueryId,
        params: D,
    ) -> (D, Vec<QueryTypeId>) {
        let ctx = QueryCtx::new(self);
        let res = ctx.get(query_id, params.into());
        let executed = ::std::mem::replace(&mut *ctx.executed.borrow_mut(), Vec::new());
        (res, executed)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Gen(u64);
const INVALIDATED: Gen = Gen(!0);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InputFingerprint(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OutputFingerprint(pub u64);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryTypeId(pub u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QueryId(pub QueryTypeId, pub InputFingerprint);

