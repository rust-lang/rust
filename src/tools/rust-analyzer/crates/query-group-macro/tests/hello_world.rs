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
    #[salsa::invoke_interned(length_query)]
    fn length_query(&self, key: ()) -> usize;

    // unadorned query
    fn length_query_with_no_params(&self) -> usize;

    // renamed/invoke query
    #[salsa::invoke_interned(invoke_length_query_actual)]
    fn invoke_length_query(&self, key: ()) -> usize;

    // not a query. should not invoked
    #[salsa::transparent]
    fn transparent_length(&self, key: ()) -> usize;

    #[salsa::transparent]
    #[salsa::invoke_interned(transparent_and_invoke_length_actual)]
    fn transparent_and_invoke_length(&self, key: ()) -> usize;
}

fn length_query(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

fn length_query_with_no_params(db: &dyn HelloWorldDatabase) -> usize {
    InputString::get(db).inner(db).len()
}

fn invoke_length_query_actual(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

fn transparent_length(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

fn transparent_and_invoke_length_actual(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

#[test]
fn unadorned_query() {
    let db = LoggerDb::default();

    InputString::new(&db, String::from("Hello, world!"));
    let len = db.length_query(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_HelloWorldDatabase(Id(400)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length_query_shim(Id(c00)) })",
        ]"#]]);
}

#[test]
fn invoke_query() {
    let db = LoggerDb::default();

    InputString::new(&db, String::from("Hello, world!"));
    let len = db.invoke_length_query(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_HelloWorldDatabase(Id(400)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: invoke_length_query_shim(Id(c00)) })",
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

#[test]
fn transparent_invoke() {
    let db = LoggerDb::default();

    InputString::new(&db, String::from("Hello, world!"));
    let len = db.transparent_and_invoke_length(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_HelloWorldDatabase(Id(400)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: transparent_and_invoke_length_shim(Id(c00)) })",
        ]"#]]);
}
