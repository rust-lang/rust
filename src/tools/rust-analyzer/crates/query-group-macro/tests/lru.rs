use expect_test::expect;

mod logger_db;
use logger_db::LoggerDb;
use query_group_macro::query_group;

#[salsa::input(singleton)]
struct InputString {
    inner: String,
}

#[query_group]
pub trait LruDB: salsa::Database {
    #[salsa::lru(16)]
    #[salsa::invoke_interned(length_query)]
    fn length_query(&self, key: ()) -> usize;

    #[salsa::lru(16)]
    #[salsa::invoke_interned(invoked_query)]
    fn length_query_invoke(&self, key: ()) -> usize;
}

fn length_query(db: &dyn LruDB, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

fn invoked_query(db: &dyn LruDB, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

#[test]
fn plain_lru() {
    let db = LoggerDb::default();

    InputString::new(&db, String::from("Hello, world!"));
    let len = db.length_query(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_LruDB(Id(400)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length_query_shim(Id(c00)) })",
        ]"#]]);
}

#[test]
fn invoke_lru() {
    let db = LoggerDb::default();

    InputString::new(&db, String::from("Hello, world!"));
    let len = db.length_query_invoke(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_LruDB(Id(400)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length_query_invoke_shim(Id(c00)) })",
        ]"#]]);
}
