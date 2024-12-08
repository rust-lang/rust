//! Test that "on-demand" input pattern works.
//!
//! On-demand inputs are inputs computed lazily on the fly. They are simulated
//! via a b query with zero inputs, which uses `add_synthetic_read` to
//! tweak durability and `invalidate` to clear the input.

#![allow(clippy::disallowed_types, clippy::type_complexity)]

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use ra_salsa::{Database as _, Durability, EventKind};

#[ra_salsa::query_group(QueryGroupStorage)]
trait QueryGroup: ra_salsa::Database + AsRef<HashMap<u32, u32>> {
    fn a(&self, x: u32) -> u32;
    fn b(&self, x: u32) -> u32;
    fn c(&self, x: u32) -> u32;
}

fn a(db: &dyn QueryGroup, x: u32) -> u32 {
    let durability = if x % 2 == 0 { Durability::LOW } else { Durability::HIGH };
    db.salsa_runtime().report_synthetic_read(durability);
    let external_state: &HashMap<u32, u32> = db.as_ref();
    external_state[&x]
}

fn b(db: &dyn QueryGroup, x: u32) -> u32 {
    db.a(x)
}

fn c(db: &dyn QueryGroup, x: u32) -> u32 {
    db.b(x)
}

#[ra_salsa::database(QueryGroupStorage)]
#[derive(Default)]
struct Database {
    storage: ra_salsa::Storage<Self>,
    external_state: HashMap<u32, u32>,
    on_event: Option<Box<dyn Fn(&Database, ra_salsa::Event)>>,
}

impl ra_salsa::Database for Database {
    fn salsa_event(&self, event: ra_salsa::Event) {
        if let Some(cb) = &self.on_event {
            cb(self, event)
        }
    }
}

impl AsRef<HashMap<u32, u32>> for Database {
    fn as_ref(&self) -> &HashMap<u32, u32> {
        &self.external_state
    }
}

#[test]
fn on_demand_input_works() {
    let mut db = Database::default();

    db.external_state.insert(1, 10);
    assert_eq!(db.b(1), 10);
    assert_eq!(db.a(1), 10);

    // We changed external state, but haven't signaled about this yet,
    // so we expect to see the old answer
    db.external_state.insert(1, 92);
    assert_eq!(db.b(1), 10);
    assert_eq!(db.a(1), 10);

    AQuery.in_db_mut(&mut db).invalidate(&1);
    assert_eq!(db.b(1), 92);
    assert_eq!(db.a(1), 92);

    // Downstream queries should also be rerun if we call `a` first.
    db.external_state.insert(1, 50);
    AQuery.in_db_mut(&mut db).invalidate(&1);
    assert_eq!(db.a(1), 50);
    assert_eq!(db.b(1), 50);
}

#[test]
fn on_demand_input_durability() {
    let mut db = Database::default();

    let events = Rc::new(RefCell::new(vec![]));
    db.on_event = Some(Box::new({
        let events = events.clone();
        move |db, event| {
            if let EventKind::WillCheckCancellation = event.kind {
                // these events are not interesting
            } else {
                events.borrow_mut().push(format!("{:?}", event.debug(db)))
            }
        }
    }));

    events.replace(vec![]);
    db.external_state.insert(1, 10);
    db.external_state.insert(2, 20);
    assert_eq!(db.b(1), 10);
    assert_eq!(db.b(2), 20);
    expect_test::expect![[r#"
        RefCell {
            value: [
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::BQuery::b(1) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::AQuery::a(1) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::BQuery::b(2) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::AQuery::a(2) } }",
            ],
        }
    "#]].assert_debug_eq(&events);

    db.synthetic_write(Durability::LOW);
    events.replace(vec![]);
    assert_eq!(db.c(1), 10);
    assert_eq!(db.c(2), 20);
    // Re-execute `a(2)` because that has low durability, but not `a(1)`
    expect_test::expect![[r#"
        RefCell {
            value: [
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::CQuery::c(1) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: DidValidateMemoizedValue { database_key: on_demand_inputs::BQuery::b(1) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::CQuery::c(2) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::AQuery::a(2) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: DidValidateMemoizedValue { database_key: on_demand_inputs::BQuery::b(2) } }",
            ],
        }
    "#]].assert_debug_eq(&events);

    db.synthetic_write(Durability::HIGH);
    events.replace(vec![]);
    assert_eq!(db.c(1), 10);
    assert_eq!(db.c(2), 20);
    // Re-execute both `a(1)` and `a(2)`, but we don't re-execute any `b` queries as the
    // result didn't actually change.
    expect_test::expect![[r#"
        RefCell {
            value: [
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::AQuery::a(1) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: DidValidateMemoizedValue { database_key: on_demand_inputs::CQuery::c(1) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: WillExecute { database_key: on_demand_inputs::AQuery::a(2) } }",
                "Event { runtime_id: RuntimeId { counter: 0 }, kind: DidValidateMemoizedValue { database_key: on_demand_inputs::CQuery::c(2) } }",
            ],
        }
    "#]].assert_debug_eq(&events);
}
