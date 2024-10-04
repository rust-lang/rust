use crate::implementation::{TestContext, TestContextImpl};

#[ra_salsa::query_group(MemoizedInputs)]
pub(crate) trait MemoizedInputsContext: TestContext {
    fn max(&self) -> usize;
    #[ra_salsa::input]
    fn input1(&self) -> usize;
    #[ra_salsa::input]
    fn input2(&self) -> usize;
}

fn max(db: &dyn MemoizedInputsContext) -> usize {
    db.log().add("Max invoked");
    std::cmp::max(db.input1(), db.input2())
}

#[test]
fn revalidate() {
    let db = &mut TestContextImpl::default();

    db.set_input1(0);
    db.set_input2(0);

    let v = db.max();
    assert_eq!(v, 0);
    db.assert_log(&["Max invoked"]);

    let v = db.max();
    assert_eq!(v, 0);
    db.assert_log(&[]);

    db.set_input1(44);
    db.assert_log(&[]);

    let v = db.max();
    assert_eq!(v, 44);
    db.assert_log(&["Max invoked"]);

    let v = db.max();
    assert_eq!(v, 44);
    db.assert_log(&[]);

    db.set_input1(44);
    db.assert_log(&[]);
    db.set_input2(66);
    db.assert_log(&[]);
    db.set_input1(64);
    db.assert_log(&[]);

    let v = db.max();
    assert_eq!(v, 66);
    db.assert_log(&["Max invoked"]);

    let v = db.max();
    assert_eq!(v, 66);
    db.assert_log(&[]);
}

/// Test that invoking `set` on an input with the same value still
/// triggers a new revision.
#[test]
fn set_after_no_change() {
    let db = &mut TestContextImpl::default();

    db.set_input2(0);

    db.set_input1(44);
    let v = db.max();
    assert_eq!(v, 44);
    db.assert_log(&["Max invoked"]);

    db.set_input1(44);
    let v = db.max();
    assert_eq!(v, 44);
    db.assert_log(&["Max invoked"]);
}
