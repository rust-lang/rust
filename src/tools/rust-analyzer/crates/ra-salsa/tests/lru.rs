//! Test setting LRU actually limits the number of things in the database;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug, PartialEq, Eq)]
struct HotPotato(u32);

static N_POTATOES: AtomicUsize = AtomicUsize::new(0);

impl HotPotato {
    fn new(id: u32) -> HotPotato {
        N_POTATOES.fetch_add(1, Ordering::SeqCst);
        HotPotato(id)
    }
}

impl Drop for HotPotato {
    fn drop(&mut self) {
        N_POTATOES.fetch_sub(1, Ordering::SeqCst);
    }
}

#[ra_salsa::query_group(QueryGroupStorage)]
trait QueryGroup: ra_salsa::Database {
    #[ra_salsa::lru]
    fn get(&self, x: u32) -> Arc<HotPotato>;
    #[ra_salsa::lru]
    fn get_volatile(&self, x: u32) -> usize;
}

fn get(_db: &dyn QueryGroup, x: u32) -> Arc<HotPotato> {
    Arc::new(HotPotato::new(x))
}

fn get_volatile(db: &dyn QueryGroup, _x: u32) -> usize {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    db.salsa_runtime().report_untracked_read();
    COUNTER.fetch_add(1, Ordering::SeqCst)
}

#[ra_salsa::database(QueryGroupStorage)]
#[derive(Default)]
struct Database {
    storage: ra_salsa::Storage<Self>,
}

impl ra_salsa::Database for Database {}

#[test]
fn lru_works() {
    let mut db = Database::default();
    GetQuery.in_db_mut(&mut db).set_lru_capacity(32);
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 0);

    for i in 0..128u32 {
        let p = db.get(i);
        assert_eq!(p.0, i)
    }
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 32);

    for i in 0..128u32 {
        let p = db.get(i);
        assert_eq!(p.0, i)
    }
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 32);

    GetQuery.in_db_mut(&mut db).set_lru_capacity(32);
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 32);

    GetQuery.in_db_mut(&mut db).set_lru_capacity(64);
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 32);
    for i in 0..128u32 {
        let p = db.get(i);
        assert_eq!(p.0, i)
    }
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 64);

    // Special case: setting capacity to zero disables LRU
    GetQuery.in_db_mut(&mut db).set_lru_capacity(0);
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 64);
    for i in 0..128u32 {
        let p = db.get(i);
        assert_eq!(p.0, i)
    }
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 128);

    drop(db);
    assert_eq!(N_POTATOES.load(Ordering::SeqCst), 0);
}

#[test]
fn lru_doesnt_break_volatile_queries() {
    let mut db = Database::default();
    GetVolatileQuery.in_db_mut(&mut db).set_lru_capacity(32);
    // Here, we check that we execute each volatile query at most once, despite
    // LRU. That does mean that we have more values in DB than the LRU capacity,
    // but it's much better than inconsistent results from volatile queries!
    for i in (0..3).flat_map(|_| 0..128usize) {
        let x = db.get_volatile(i as u32);
        assert_eq!(x, i)
    }
}
