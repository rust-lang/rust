use query_group_macro::query_group;

mod logger_db;
use expect_test::expect;
use logger_db::LoggerDb;

#[query_group]
pub trait HelloWorldDatabase: salsa::Database {
    #[salsa::input]
    fn input_string(&self) -> String;

    #[salsa::invoke_interned(length_query)]
    fn length_query(&self, key: ()) -> (usize, usize);
}

fn length_query(db: &dyn HelloWorldDatabase, _key: ()) -> (usize, usize) {
    let len = db.input_string().len();
    (len, len)
}

#[test]
fn query() {
    let mut db = LoggerDb::default();

    db.set_input_string(String::from("Hello, world!"));
    let len = db.length_query(());

    assert_eq!(len, (13, 13));
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
