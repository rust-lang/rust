use rand::seq::SliceRandom;
use rand::Rng;

use ra_salsa::ParallelDatabase;
use ra_salsa::Snapshot;
use ra_salsa::{Cancelled, Database};

// Number of operations a reader performs
const N_MUTATOR_OPS: usize = 100;
const N_READER_OPS: usize = 100;

#[ra_salsa::query_group(Stress)]
trait StressDatabase: ra_salsa::Database {
    #[ra_salsa::input]
    fn a(&self, key: usize) -> usize;

    fn b(&self, key: usize) -> usize;

    fn c(&self, key: usize) -> usize;
}

fn b(db: &dyn StressDatabase, key: usize) -> usize {
    db.unwind_if_cancelled();
    db.a(key)
}

fn c(db: &dyn StressDatabase, key: usize) -> usize {
    db.b(key)
}

#[ra_salsa::database(Stress)]
#[derive(Default)]
struct StressDatabaseImpl {
    storage: ra_salsa::Storage<Self>,
}

impl ra_salsa::Database for StressDatabaseImpl {}

impl ra_salsa::ParallelDatabase for StressDatabaseImpl {
    fn snapshot(&self) -> Snapshot<StressDatabaseImpl> {
        Snapshot::new(StressDatabaseImpl { storage: self.storage.snapshot() })
    }
}

#[derive(Clone, Copy, Debug)]
enum Query {
    A,
    B,
    C,
}

enum MutatorOp {
    WriteOp(WriteOp),
    LaunchReader { ops: Vec<ReadOp>, check_cancellation: bool },
}

#[derive(Debug)]
enum WriteOp {
    SetA(usize, usize),
}

#[derive(Debug)]
enum ReadOp {
    Get(Query, usize),
}

impl rand::distributions::Distribution<Query> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Query {
        *[Query::A, Query::B, Query::C].choose(rng).unwrap()
    }
}

impl rand::distributions::Distribution<MutatorOp> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> MutatorOp {
        if rng.gen_bool(0.5) {
            MutatorOp::WriteOp(rng.gen())
        } else {
            MutatorOp::LaunchReader {
                ops: (0..N_READER_OPS).map(|_| rng.gen()).collect(),
                check_cancellation: rng.gen(),
            }
        }
    }
}

impl rand::distributions::Distribution<WriteOp> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> WriteOp {
        let key = rng.gen::<usize>() % 10;
        let value = rng.gen::<usize>() % 10;
        WriteOp::SetA(key, value)
    }
}

impl rand::distributions::Distribution<ReadOp> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> ReadOp {
        let query = rng.gen::<Query>();
        let key = rng.gen::<usize>() % 10;
        ReadOp::Get(query, key)
    }
}

fn db_reader_thread(db: &StressDatabaseImpl, ops: Vec<ReadOp>, check_cancellation: bool) {
    for op in ops {
        if check_cancellation {
            db.unwind_if_cancelled();
        }
        op.execute(db);
    }
}

impl WriteOp {
    fn execute(self, db: &mut StressDatabaseImpl) {
        match self {
            WriteOp::SetA(key, value) => {
                db.set_a(key, value);
            }
        }
    }
}

impl ReadOp {
    fn execute(self, db: &StressDatabaseImpl) {
        match self {
            ReadOp::Get(query, key) => match query {
                Query::A => {
                    db.a(key);
                }
                Query::B => {
                    let _ = db.b(key);
                }
                Query::C => {
                    let _ = db.c(key);
                }
            },
        }
    }
}

#[test]
fn stress_test() {
    let mut db = StressDatabaseImpl::default();
    for i in 0..10 {
        db.set_a(i, i);
    }

    let mut rng = rand::thread_rng();

    // generate the ops that the mutator thread will perform
    let write_ops: Vec<MutatorOp> = (0..N_MUTATOR_OPS).map(|_| rng.gen()).collect();

    // execute the "main thread", which sometimes snapshots off other threads
    let mut all_threads = vec![];
    for op in write_ops {
        match op {
            MutatorOp::WriteOp(w) => w.execute(&mut db),
            MutatorOp::LaunchReader { ops, check_cancellation } => {
                all_threads.push(std::thread::spawn({
                    let db = db.snapshot();
                    move || Cancelled::catch(|| db_reader_thread(&db, ops, check_cancellation))
                }))
            }
        }
    }

    for thread in all_threads {
        thread.join().unwrap().ok();
    }
}
