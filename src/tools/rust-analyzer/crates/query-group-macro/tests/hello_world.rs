use expect_test::expect;
use query_group_macro::query_group;

mod logger_db;
use logger_db::LoggerDb;

#[query_group]
pub trait HelloWorldDatabase: salsa::Database {
    // input
    // // input with no params
    #[salsa::input]
    fn input_string(&self) -> String;

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
    db.input_string().len()
}

fn length_query_with_no_params(db: &dyn HelloWorldDatabase) -> usize {
    db.input_string().len()
}

fn invoke_length_query_actual(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    db.input_string().len()
}

fn transparent_length(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    db.input_string().len()
}

fn transparent_and_invoke_length_actual(db: &dyn HelloWorldDatabase, _key: ()) -> usize {
    db.input_string().len()
}

#[test]
fn unadorned_query() {
    let mut db = LoggerDb::default();

    db.set_input_string(String::from("Hello, world!"));
    let len = db.length_query(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_HelloWorldDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(DidValidateMemoizedValue { database_key: create_data_HelloWorldDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length_query_shim(Id(800)) })",
            "salsa_event(WillCheckCancellation)",
        ]"#]]);
}

#[test]
fn invoke_query() {
    let mut db = LoggerDb::default();

    db.set_input_string(String::from("Hello, world!"));
    let len = db.invoke_length_query(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_HelloWorldDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(DidValidateMemoizedValue { database_key: create_data_HelloWorldDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: invoke_length_query_shim(Id(800)) })",
            "salsa_event(WillCheckCancellation)",
        ]"#]]);
}

#[test]
fn transparent() {
    let mut db = LoggerDb::default();

    db.set_input_string(String::from("Hello, world!"));
    let len = db.transparent_length(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_HelloWorldDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(DidValidateMemoizedValue { database_key: create_data_HelloWorldDatabase(Id(0)) })",
        ]"#]]);
}

#[test]
fn transparent_invoke() {
    let mut db = LoggerDb::default();

    db.set_input_string(String::from("Hello, world!"));
    let len = db.transparent_and_invoke_length(());

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_HelloWorldDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(DidValidateMemoizedValue { database_key: create_data_HelloWorldDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: transparent_and_invoke_length_shim(Id(800)) })",
            "salsa_event(WillCheckCancellation)",
        ]"#]]);
}
