mod logger_db;
use expect_test::expect;
use logger_db::LoggerDb;

use query_group_macro::query_group;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error;

#[query_group]
pub trait ResultDatabase: salsa::Database {
    #[salsa::input]
    fn input_string(&self) -> String;

    #[salsa::invoke_interned(length)]
    fn length(&self, key: ()) -> Result<usize, Error>;

    #[salsa::invoke_interned(length2)]
    fn length2(&self, key: ()) -> Result<usize, Error>;
}

fn length(db: &dyn ResultDatabase, _key: ()) -> Result<usize, Error> {
    Ok(db.input_string().len())
}

fn length2(db: &dyn ResultDatabase, _key: ()) -> Result<usize, Error> {
    Ok(db.input_string().len())
}

#[test]
fn test_queries_with_results() {
    let mut db = LoggerDb::default();
    let input = "hello";
    db.set_input_string(input.to_owned());
    assert_eq!(db.length(()), Ok(input.len()));
    assert_eq!(db.length2(()), Ok(input.len()));

    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: create_data_ResultDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(DidValidateMemoizedValue { database_key: create_data_ResultDatabase(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length_shim(Id(800)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length2_shim(Id(c00)) })",
            "salsa_event(WillCheckCancellation)",
        ]"#]]);
}
