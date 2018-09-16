mod imp;

use std::{
    sync::Arc,
};
use im;
use salsa;
use {FileId, imp::FileResolverImp};

#[derive(Debug, Default, Clone)]
pub(crate) struct State {
    pub(crate) file_map: im::HashMap<FileId, Arc<String>>,
    pub(crate) file_resolver: FileResolverImp
}

#[derive(Debug)]
pub(crate) struct Db {
    imp: imp::Db,
}

#[derive(Clone, Copy)]
pub(crate) struct QueryCtx<'a> {
    imp: &'a salsa::QueryCtx<State, imp::Data>,
}

pub(crate) struct Query<T, R>(pub(crate) u16, pub(crate) fn(QueryCtx, &T) -> R);

pub(crate) struct QueryRegistry {
    imp: imp::QueryRegistry,
}

impl Default for Db {
    fn default() -> Db {
        Db::new()
    }
}

impl Db {
    pub(crate) fn new() -> Db {
        let reg = QueryRegistry::new();
        Db { imp: imp::Db::new(reg.imp) }
    }
    pub(crate) fn state(&self) -> &State {
        self.imp.imp.ground_data()
    }
    pub(crate) fn with_changes(&self, new_state: State, changed_files: &[FileId], resolver_changed: bool) -> Db {
        Db { imp: self.imp.with_changes(new_state, changed_files, resolver_changed) }
    }
    pub(crate) fn make_query<F: FnOnce(QueryCtx) -> R, R>(&self, f: F) -> R {
        let ctx = QueryCtx { imp: &self.imp.imp.query_ctx() };
        f(ctx)
    }
    #[allow(unused)]
    pub(crate) fn trace_query<F: FnOnce(QueryCtx) -> R, R>(&self, f: F) -> (R, Vec<&'static str>) {
        let ctx = QueryCtx { imp: &self.imp.imp.query_ctx() };
        let res = f(ctx);
        let trace = self.imp.extract_trace(ctx.imp);
        (res, trace)
    }
}

impl<'a> QueryCtx<'a> {
    pub(crate) fn get<Q: imp::EvalQuery>(&self, q: Q, params: Q::Params) -> Arc<Q::Output> {
        q.get(self, params)
    }
}

pub(crate) fn file_text(ctx: QueryCtx, file_id: FileId) -> Arc<String> {
    imp::file_text(ctx, file_id)
}

pub(crate) fn file_set(ctx: QueryCtx) -> Arc<(Vec<FileId>, FileResolverImp)> {
    imp::file_set(ctx)
}
impl QueryRegistry {
    fn new() -> QueryRegistry {
        let mut reg = QueryRegistry { imp: imp::QueryRegistry::new() };
        ::queries::register_queries(&mut reg);
        ::module_map::register_queries(&mut reg);
        reg
    }
    pub(crate) fn add<Q: imp::EvalQuery>(&mut self, q: Q, name: &'static str) {
        self.imp.add(q, name)
    }
}
