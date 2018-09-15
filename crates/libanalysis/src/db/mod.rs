mod queries;

use std::{
    hash::{Hash},
    sync::Arc,
    fmt::Debug,
    any::Any,
    iter,
};
use im;
use salsa;
use {
    FileId,
    imp::{FileResolverImp},
};


#[derive(Clone, Default)]
pub(crate) struct State {
    pub(crate) resolver: FileResolverImp,
    pub(crate) file_map: im::HashMap<FileId, Arc<str>>,
}

type Data = Arc<Any + Send + Sync + 'static>;

pub(crate) struct QueryCtx<'a> {
    inner: &'a salsa::QueryCtx<State, Data>
}

pub(crate) struct Db {
    inner: salsa::Db<State, Data>
}

struct GroundQuery<T, R> {
    id: u16,
    f: fn(&State, &T) -> R,
    h: fn(&R) -> u64,
}

pub(crate) struct Query<T, R> {
    pub(crate) id: u16,
    pub(crate) f: fn(QueryCtx, &T) -> R,
}

impl Db {
    pub(crate) fn new(state: State) -> Db {
        Db { inner: salsa::Db::new(query_config(), state) }
    }
    pub(crate) fn state(&self) -> &State {
        self.inner.ground_data()
    }
    pub(crate) fn with_state(
        &self,
        new_state: State,
        updated_files: &[FileId],
        file_set_changed: bool,
    ) -> Db {
        let mut inv = salsa::Invalidations::new();
        if file_set_changed {
            inv.invalidate(
                salsa::QueryTypeId(queries::FILE_SET.id),
                iter::once(salsa::InputFingerprint(hash(&()))),
            );
        } else {
            inv.invalidate(
                salsa::QueryTypeId(queries::FILE_SET.id),
                iter::empty(),
            );
        }
        inv.invalidate(
            salsa::QueryTypeId(queries::FILE_TEXT.id),
            updated_files.iter().map(hash).map(salsa::InputFingerprint),
        );
        Db { inner: self.inner.with_ground_data(new_state, inv) }
    }
    pub(crate) fn get<T, R>(&self, q: Query<T, R>, params: T) -> (Arc<R>, Vec<u16>)
    where
        T: Hash + Send + Sync + 'static,
        R: Send + Sync + 'static,
    {
        let query_id = salsa::QueryId(
            salsa::QueryTypeId(q.id),
            salsa::InputFingerprint(hash(&params)),
        );
        let params = Arc::new(params);
        let (res, events) = self.inner.get(query_id, params);
        let res = res.downcast().unwrap();
        let events = events.into_iter().map(|it| it.0).collect();
        (res, events)
    }

}

impl<'a> QueryCtx<'a> {
    fn get_g<T, R>(&self, q: GroundQuery<T, R>, params: T) -> Arc<R>
    where
        T: Hash + Send + Sync + 'static,
        R: Send + Sync + 'static,
     {
        let query_id = salsa::QueryId(
            salsa::QueryTypeId(q.id),
            salsa::InputFingerprint(hash(&params)),
        );
        let res = self.inner.get(query_id, Arc::new(params));
        res.downcast().unwrap()
    }
    pub(crate) fn get<T, R>(&self, q: Query<T, R>, params: T) -> Arc<R>
    where
        T: Hash + Send + Sync + 'static,
        R: Send + Sync + 'static,
     {
        let query_id = salsa::QueryId(
            salsa::QueryTypeId(q.id),
            salsa::InputFingerprint(hash(&params)),
        );
        let res = self.inner.get(query_id, Arc::new(params));
        res.downcast().unwrap()
    }
}

fn query_config() -> salsa::QueryConfig<State, Data> {
    let mut res = salsa::QueryConfig::new();
    let queries: Vec<BoxedGroundQuery> = vec![
        queries::FILE_TEXT.into(),
        queries::FILE_SET.into(),
    ];
    for q in queries {
        res = res.with_ground_query(q.query_type, q.f)
    }
    let mut queries: Vec<BoxedQuery> = vec![
        queries::FILE_SYNTAX.into(),
    ];
    ::module_map_db::queries(&mut queries);
    for q in queries {
        res = res.with_query(q.query_type, q.f);
    }
    res
}

struct BoxedGroundQuery {
    query_type: salsa::QueryTypeId,
    f: Box<Fn(&State, &Data) -> (Data, salsa::OutputFingerprint) + Send + Sync + 'static>,
}

impl<T, R> From<GroundQuery<T, R>> for BoxedGroundQuery
where
    T: Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    fn from(q: GroundQuery<T, R>) -> BoxedGroundQuery
    {
        BoxedGroundQuery {
            query_type: salsa::QueryTypeId(q.id),
            f: Box::new(move |state, data| {
                let data: &T = data.downcast_ref().unwrap();
                let res = (q.f)(state, data);
                let h = (q.h)(&res);
                (Arc::new(res), salsa::OutputFingerprint(h))
            })
        }
    }
}

pub(crate) struct BoxedQuery {
    query_type: salsa::QueryTypeId,
    f: Box<Fn(&salsa::QueryCtx<State, Data>, &Data) -> (Data, salsa::OutputFingerprint) + Send + Sync + 'static>,
}

impl<T, R> From<Query<T, R>> for BoxedQuery
where
    T: Hash + Send + Sync + 'static,
    R: Hash + Send + Sync + 'static,
{
    fn from(q: Query<T, R>) -> BoxedQuery
    {
        BoxedQuery {
            query_type: salsa::QueryTypeId(q.id),
            f: Box::new(move |ctx, data| {
                let ctx = QueryCtx { inner: ctx };
                let data: &T = data.downcast_ref().unwrap();
                let res = (q.f)(ctx, data);
                let h = hash(&res);
                (Arc::new(res), salsa::OutputFingerprint(h))
            })
        }
    }
}

fn hash<T: ::std::hash::Hash>(x: &T) -> u64 {
    use std::hash::Hasher;
    let mut hasher = ::std::collections::hash_map::DefaultHasher::new();
    ::std::hash::Hash::hash(x, &mut hasher);
    hasher.finish()
}
