use hir_def::db::DefDatabase;
use rustc_hash::FxHashMap;
use span::Edition;
use test_fixture::WithFixture;
use triomphe::Arc;

use crate::{
    db::HirDatabase,
    mir::{MirBody, MirLowerError},
    setup_tracing,
    test_db::TestDB,
};

fn lower_mir(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
) -> FxHashMap<String, Result<Arc<MirBody>, MirLowerError>> {
    let _tracing = setup_tracing();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    let file_id = *file_ids.last().unwrap();
    let module_id = db.module_for_file(file_id.file_id(&db));
    let def_map = module_id.def_map(&db);
    let scope = &def_map[module_id.local_id].scope;
    let funcs = scope.declarations().filter_map(|x| match x {
        hir_def::ModuleDefId::FunctionId(it) => Some(it),
        _ => None,
    });
    funcs
        .map(|func| {
            let name = db.function_signature(func).name.display(&db, Edition::CURRENT).to_string();
            let mir = db.mir_body(func.into());
            (name, mir)
        })
        .collect()
}

#[test]
fn dyn_projection_with_auto_traits_regression_next_solver() {
    lower_mir(
        r#"
//- minicore: sized, send
pub trait Deserializer {}

pub trait Strictest {
    type Object: ?Sized;
}

impl Strictest for dyn CustomValue {
    type Object = dyn CustomValue + Send;
}

pub trait CustomValue: Send {}

impl CustomValue for () {}

struct Box<T: ?Sized>;

type DeserializeFn<T> = fn(&mut dyn Deserializer) -> Box<T>;

fn foo() {
    (|deserializer| Box::new(())) as DeserializeFn<<dyn CustomValue as Strictest>::Object>;
}
    "#,
    );
}
