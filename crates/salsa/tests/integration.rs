extern crate salsa;
use std::{
    sync::Arc,
    collections::hash_map::{HashMap, DefaultHasher},
    any::Any,
    hash::{Hash, Hasher},
};

type State = HashMap<u32, String>;
const GET_TEXT: salsa::QueryTypeId = salsa::QueryTypeId(1);
const GET_FILES: salsa::QueryTypeId = salsa::QueryTypeId(2);
const FILE_NEWLINES: salsa::QueryTypeId = salsa::QueryTypeId(3);
const TOTAL_NEWLINES: salsa::QueryTypeId = salsa::QueryTypeId(4);

fn mk_ground_query<T, R>(
    state: &State,
    params: &(Any + Send + Sync + 'static),
    f: fn(&State, &T) -> R,
) -> (Box<Any + Send + Sync + 'static>, salsa::OutputFingerprint)
where
    T: 'static,
    R: Hash + Send + Sync + 'static,
{
    let params = params.downcast_ref().unwrap();
    let result = f(state, params);
    let fingerprint = o_print(&result);
    (Box::new(result), fingerprint)
}

fn get<T, R>(db: &salsa::Db<State>, query_type: salsa::QueryTypeId, param: T) -> (Arc<R>, Vec<salsa::QueryTypeId>)
where
    T: Hash + Send + Sync + 'static,
    R: Send + Sync + 'static,
{
    let i_print = i_print(&param);
    let param = Box::new(param);
    let (res, trace) = db.get(salsa::QueryId(query_type, i_print), param);
    (res.downcast().unwrap(), trace)
}

struct QueryCtx<'a>(&'a salsa::QueryCtx<State>);

impl<'a> QueryCtx<'a> {
    fn get_text(&self, id: u32) -> Arc<String> {
        let i_print = i_print(&id);
        let text = self.0.get(salsa::QueryId(GET_TEXT, i_print), Arc::new(id));
        text.downcast().unwrap()
    }
    fn get_files(&self) -> Arc<Vec<u32>> {
        let i_print = i_print(&());
        let files = self.0.get(salsa::QueryId(GET_FILES, i_print), Arc::new(()));
        let res = files.downcast().unwrap();
        res
    }
    fn get_n_lines(&self, id: u32) -> usize {
        let i_print = i_print(&id);
        let n_lines = self.0.get(salsa::QueryId(FILE_NEWLINES, i_print), Arc::new(id));
        *n_lines.downcast().unwrap()
    }
}

fn mk_query<T, R>(
    query_ctx: &salsa::QueryCtx<State>,
    params: &(Any + Send + Sync + 'static),
    f: fn(QueryCtx, &T) -> R,
) -> (Box<Any + Send + Sync + 'static>, salsa::OutputFingerprint)
where
    T: 'static,
    R: Hash + Send + Sync + 'static,
{
    let params: &T = params.downcast_ref().unwrap();
    let query_ctx = QueryCtx(query_ctx);
    let result = f(query_ctx, params);
    let fingerprint = o_print(&result);
    (Box::new(result), fingerprint)
}

fn mk_queries() -> salsa::QueryConfig<State> {
    salsa::QueryConfig::<State>::new()
        .with_ground_query(GET_TEXT, |state, id| {
            mk_ground_query::<u32, String>(state, id, |state, id| state[id].clone())
        })
        .with_ground_query(GET_FILES, |state, id| {
            mk_ground_query::<(), Vec<u32>>(state, id, |state, &()| state.keys().cloned().collect())
        })
        .with_query(FILE_NEWLINES, |query_ctx, id| {
            mk_query(query_ctx, id, |query_ctx, &id| {
                let text = query_ctx.get_text(id);
                text.lines().count()
            })
        })
        .with_query(TOTAL_NEWLINES, |query_ctx, id| {
            mk_query(query_ctx, id, |query_ctx, &()| {
                let mut total = 0;
                for &id in query_ctx.get_files().iter() {
                    total += query_ctx.get_n_lines(id)
                }
                total
            })
        })
}

#[test]
fn test_number_of_lines() {
    let mut state = State::new();
    let db = salsa::Db::new(mk_queries(), state.clone());
    let (newlines, trace) = get::<(), usize>(&db, TOTAL_NEWLINES, ());
    assert_eq!(*newlines, 0);
    assert_eq!(trace.len(), 2);
    let (newlines, trace) = get::<(), usize>(&db, TOTAL_NEWLINES, ());
    assert_eq!(*newlines, 0);
    assert_eq!(trace.len(), 0);

    state.insert(1, "hello\nworld".to_string());
    let db = db.with_ground_data(state.clone());
    let (newlines, trace) = get::<(), usize>(&db, TOTAL_NEWLINES, ());
    assert_eq!(*newlines, 2);
    assert_eq!(trace.len(), 4);

    state.insert(2, "spam\neggs".to_string());
    let db = db.with_ground_data(state.clone());
    let (newlines, trace) = get::<(), usize>(&db, TOTAL_NEWLINES, ());
    assert_eq!(*newlines, 4);
    assert_eq!(trace.len(), 5);

    for i in 0..10 {
        state.insert(i + 10, "spam".to_string());
    }
    let db = db.with_ground_data(state.clone());
    let (newlines, trace) = get::<(), usize>(&db, TOTAL_NEWLINES, ());
    assert_eq!(*newlines, 14);
    assert_eq!(trace.len(), 24);

    state.insert(15, String::new());
    let db = db.with_ground_data(state.clone());
    let (newlines, trace) = get::<(), usize>(&db, TOTAL_NEWLINES, ());
    assert_eq!(*newlines, 13);
    assert_eq!(trace.len(), 15);
}

fn o_print<T: Hash>(x: &T) -> salsa::OutputFingerprint {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    let hash = hasher.finish();
    salsa::OutputFingerprint(hash)
}

fn i_print<T: Hash>(x: &T) -> salsa::InputFingerprint {
    let mut hasher = DefaultHasher::new();
    x.hash(&mut hasher);
    let hash = hasher.finish();
    salsa::InputFingerprint(hash)
}
