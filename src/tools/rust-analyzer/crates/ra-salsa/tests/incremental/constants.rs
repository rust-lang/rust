use crate::implementation::{TestContext, TestContextImpl};
use ra_salsa::debug::DebugQueryTable;
use ra_salsa::Durability;

#[ra_salsa::query_group(Constants)]
pub(crate) trait ConstantsDatabase: TestContext {
    #[ra_salsa::input]
    fn input(&self, key: char) -> usize;

    fn add(&self, key1: char, key2: char) -> usize;

    fn add3(&self, key1: char, key2: char, key3: char) -> usize;
}

fn add(db: &dyn ConstantsDatabase, key1: char, key2: char) -> usize {
    db.log().add(format!("add({key1}, {key2})"));
    db.input(key1) + db.input(key2)
}

fn add3(db: &dyn ConstantsDatabase, key1: char, key2: char, key3: char) -> usize {
    db.log().add(format!("add3({key1}, {key2}, {key3})"));
    db.add(key1, key2) + db.input(key3)
}

// Test we can assign a constant and things will be correctly
// recomputed afterwards.
#[test]
fn invalidate_constant() {
    let db = &mut TestContextImpl::default();
    db.set_input_with_durability('a', 44, Durability::HIGH);
    db.set_input_with_durability('b', 22, Durability::HIGH);
    assert_eq!(db.add('a', 'b'), 66);

    db.set_input_with_durability('a', 66, Durability::HIGH);
    assert_eq!(db.add('a', 'b'), 88);
}

#[test]
fn invalidate_constant_1() {
    let db = &mut TestContextImpl::default();

    // Not constant:
    db.set_input('a', 44);
    assert_eq!(db.add('a', 'a'), 88);

    // Becomes constant:
    db.set_input_with_durability('a', 44, Durability::HIGH);
    assert_eq!(db.add('a', 'a'), 88);

    // Invalidates:
    db.set_input_with_durability('a', 33, Durability::HIGH);
    assert_eq!(db.add('a', 'a'), 66);
}

// Test cases where we assign same value to 'a' after declaring it a
// constant.
#[test]
fn set_after_constant_same_value() {
    let db = &mut TestContextImpl::default();
    db.set_input_with_durability('a', 44, Durability::HIGH);
    db.set_input_with_durability('a', 44, Durability::HIGH);
    db.set_input('a', 44);
}

#[test]
fn not_constant() {
    let mut db = TestContextImpl::default();

    db.set_input('a', 22);
    db.set_input('b', 44);
    assert_eq!(db.add('a', 'b'), 66);
    assert_eq!(Durability::LOW, AddQuery.in_db(&db).durability(('a', 'b')));
}

#[test]
fn durability() {
    let mut db = TestContextImpl::default();

    db.set_input_with_durability('a', 22, Durability::HIGH);
    db.set_input_with_durability('b', 44, Durability::HIGH);
    assert_eq!(db.add('a', 'b'), 66);
    assert_eq!(Durability::HIGH, AddQuery.in_db(&db).durability(('a', 'b')));
}

#[test]
fn mixed_constant() {
    let mut db = TestContextImpl::default();

    db.set_input_with_durability('a', 22, Durability::HIGH);
    db.set_input('b', 44);
    assert_eq!(db.add('a', 'b'), 66);
    assert_eq!(Durability::LOW, AddQuery.in_db(&db).durability(('a', 'b')));
}

#[test]
fn becomes_constant_with_change() {
    let mut db = TestContextImpl::default();

    db.set_input('a', 22);
    db.set_input('b', 44);
    assert_eq!(db.add('a', 'b'), 66);
    assert_eq!(Durability::LOW, AddQuery.in_db(&db).durability(('a', 'b')));

    db.set_input_with_durability('a', 23, Durability::HIGH);
    assert_eq!(db.add('a', 'b'), 67);
    assert_eq!(Durability::LOW, AddQuery.in_db(&db).durability(('a', 'b')));

    db.set_input_with_durability('b', 45, Durability::HIGH);
    assert_eq!(db.add('a', 'b'), 68);
    assert_eq!(Durability::HIGH, AddQuery.in_db(&db).durability(('a', 'b')));

    db.set_input_with_durability('b', 45, Durability::MEDIUM);
    assert_eq!(db.add('a', 'b'), 68);
    assert_eq!(Durability::MEDIUM, AddQuery.in_db(&db).durability(('a', 'b')));
}

// Test a subtle case in which an input changes from constant to
// non-constant, but its value doesn't change. If we're not careful,
// this can cause us to incorrectly consider derived values as still
// being constant.
#[test]
fn constant_to_non_constant() {
    let mut db = TestContextImpl::default();

    db.set_input_with_durability('a', 11, Durability::HIGH);
    db.set_input_with_durability('b', 22, Durability::HIGH);
    db.set_input_with_durability('c', 33, Durability::HIGH);

    // Here, `add3` invokes `add`, which yields 33. Both calls are
    // constant.
    assert_eq!(db.add3('a', 'b', 'c'), 66);

    db.set_input('a', 11);

    // Here, `add3` invokes `add`, which *still* yields 33, but which
    // is no longer constant. Since value didn't change, we might
    // preserve `add3` unchanged, not noticing that it is no longer
    // constant.
    assert_eq!(db.add3('a', 'b', 'c'), 66);

    // In that case, we would not get the correct result here, when
    // 'a' changes *again*.
    db.set_input('a', 22);
    assert_eq!(db.add3('a', 'b', 'c'), 77);
}
