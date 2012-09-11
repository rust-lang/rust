#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];
/// A dynamic, mutable location.
///
/// Similar to a mutable option type, but friendlier.

struct Cell<T> {
    mut value: Option<T>
}

/// Creates a new full cell with the given value.
fn Cell<T>(+value: T) -> Cell<T> {
    Cell { value: Some(move value) }
}

fn empty_cell<T>() -> Cell<T> {
    Cell { value: None }
}

impl<T> Cell<T> {
    /// Yields the value, failing if the cell is empty.
    fn take() -> T {
        if self.is_empty() {
            fail ~"attempt to take an empty cell";
        }

        let mut value = None;
        value <-> self.value;
        return option::unwrap(value);
    }

    /// Returns the value, failing if the cell is full.
    fn put_back(+value: T) {
        if !self.is_empty() {
            fail ~"attempt to put a value back into a full cell";
        }
        self.value = Some(move value);
    }

    /// Returns true if the cell is empty and false if the cell is full.
    fn is_empty() -> bool {
        self.value.is_none()
    }

    // Calls a closure with a reference to the value.
    fn with_ref<R>(op: fn(v: &T) -> R) -> R {
        let v = self.take();
        let r = op(&v);
        self.put_back(move v);
        move r
    }
}

#[test]
fn test_basic() {
    let value_cell = Cell(~10);
    assert !value_cell.is_empty();
    let value = value_cell.take();
    assert value == ~10;
    assert value_cell.is_empty();
    value_cell.put_back(value);
    assert !value_cell.is_empty();
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_take_empty() {
    let value_cell = empty_cell::<~int>();
    value_cell.take();
}

#[test]
#[should_fail]
#[ignore(cfg(windows))]
fn test_put_back_non_empty() {
    let value_cell = Cell(~10);
    value_cell.put_back(~20);
}
