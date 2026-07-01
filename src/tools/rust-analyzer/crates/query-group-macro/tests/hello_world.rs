use expect_test::expect;
use query_group_macro::query_group;

mod logger_db;
use logger_db::LoggerDb;

#[salsa::input(singleton)]
struct InputString {
    inner: String,
}

#[query_group]
pub trait HelloWorldDatabase: salsa::Database {
    // unadorned query
    fn length_query_with_no_params(&self) -> usize;

    // not a query. should not invoked
    #[salsa::transparent]
    fn transparent_length(&self, key: ()) -> usize;
}

fn length_query_with_no_params(db: &dyn HelloWorldDatabase) -> usize {
    InputString::get(db).inner(db).len()
}

fn transparent_length(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

#[test]
fn unadorned_query() {
    let db = LoggerDb::default();

    InputString::new(&db, String::from("Hello, world!"));
    let len = db.length_query_with_no_params();

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length_query_with_no_params_shim(Id(400)) })",
        ]"#]]);
}

#[test]
fn transparent() {
    let db = LoggerDb::default();

    InputString::new(&db, String::from("Hello, world!"));
    let len = db.transparent_length(());

    assert_eq!(len, 13);
    db.assert_logs(expect!["[]"]);
}
