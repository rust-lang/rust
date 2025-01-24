use std::rc::Rc;

#[ra_salsa::query_group(NoSendSyncStorage)]
trait NoSendSyncDatabase: ra_salsa::Database {
    fn no_send_sync_value(&self, key: bool) -> Rc<bool>;
    fn no_send_sync_key(&self, key: Rc<bool>) -> bool;
}

fn no_send_sync_value(_db: &dyn NoSendSyncDatabase, key: bool) -> Rc<bool> {
    Rc::new(key)
}

fn no_send_sync_key(_db: &dyn NoSendSyncDatabase, key: Rc<bool>) -> bool {
    *key
}

#[ra_salsa::database(NoSendSyncStorage)]
#[derive(Default)]
struct DatabaseImpl {
    storage: ra_salsa::Storage<Self>,
}

impl ra_salsa::Database for DatabaseImpl {}

#[test]
fn no_send_sync() {
    let db = DatabaseImpl::default();

    assert_eq!(db.no_send_sync_value(true), Rc::new(true));
    assert!(!db.no_send_sync_key(Rc::new(false)));
}
