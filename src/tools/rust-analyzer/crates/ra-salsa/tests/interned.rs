//! Test that you can implement a query using a `dyn Trait` setup.

use ra_salsa::InternId;

#[ra_salsa::database(InternStorage)]
#[derive(Default)]
struct Database {
    storage: ra_salsa::Storage<Self>,
}

impl ra_salsa::Database for Database {}

impl ra_salsa::ParallelDatabase for Database {
    fn snapshot(&self) -> ra_salsa::Snapshot<Self> {
        ra_salsa::Snapshot::new(Database { storage: self.storage.snapshot() })
    }
}

#[ra_salsa::query_group(InternStorage)]
trait Intern {
    #[ra_salsa::interned]
    fn intern1(&self, x: String) -> InternId;

    #[ra_salsa::interned]
    fn intern2(&self, x: String, y: String) -> InternId;

    #[ra_salsa::interned]
    fn intern_key(&self, x: String) -> InternKey;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct InternKey(InternId);

impl ra_salsa::InternKey for InternKey {
    fn from_intern_id(v: InternId) -> Self {
        InternKey(v)
    }

    fn as_intern_id(&self) -> InternId {
        self.0
    }
}

#[test]
fn test_intern1() {
    let db = Database::default();
    let foo0 = db.intern1("foo".to_owned());
    let bar0 = db.intern1("bar".to_owned());
    let foo1 = db.intern1("foo".to_owned());
    let bar1 = db.intern1("bar".to_owned());

    assert_eq!(foo0, foo1);
    assert_eq!(bar0, bar1);
    assert_ne!(foo0, bar0);

    assert_eq!("foo".to_owned(), db.lookup_intern1(foo0));
    assert_eq!("bar".to_owned(), db.lookup_intern1(bar0));
}

#[test]
fn test_intern2() {
    let db = Database::default();
    let foo0 = db.intern2("x".to_owned(), "foo".to_owned());
    let bar0 = db.intern2("x".to_owned(), "bar".to_owned());
    let foo1 = db.intern2("x".to_owned(), "foo".to_owned());
    let bar1 = db.intern2("x".to_owned(), "bar".to_owned());

    assert_eq!(foo0, foo1);
    assert_eq!(bar0, bar1);
    assert_ne!(foo0, bar0);

    assert_eq!(("x".to_owned(), "foo".to_owned()), db.lookup_intern2(foo0));
    assert_eq!(("x".to_owned(), "bar".to_owned()), db.lookup_intern2(bar0));
}

#[test]
fn test_intern_key() {
    let db = Database::default();
    let foo0 = db.intern_key("foo".to_owned());
    let bar0 = db.intern_key("bar".to_owned());
    let foo1 = db.intern_key("foo".to_owned());
    let bar1 = db.intern_key("bar".to_owned());

    assert_eq!(foo0, foo1);
    assert_eq!(bar0, bar1);
    assert_ne!(foo0, bar0);

    assert_eq!("foo".to_owned(), db.lookup_intern_key(foo0));
    assert_eq!("bar".to_owned(), db.lookup_intern_key(bar0));
}
