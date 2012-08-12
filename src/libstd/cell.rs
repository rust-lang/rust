/// A dynamic, mutable location.
///
/// Similar to a mutable option type, but friendlier.

struct Cell<T> {
    mut value: option<T>;
}

/// Creates a new full cell with the given value.
fn Cell<T>(+value: T) -> Cell<T> {
    Cell { value: some(move value) }
}

fn empty_cell<T>() -> Cell<T> {
    Cell { value: none }
}

impl<T> Cell<T> {
    /// Yields the value, failing if the cell is empty.
    fn take() -> T {
        let mut value = none;
        value <-> self.value;
        if value.is_none() {
            fail ~"attempt to take an empty cell";
        }
        return option::unwrap(value);
    }

    /// Returns the value, failing if the cell is full.
    fn put_back(+value: T) {
        if self.value.is_none() {
            fail ~"attempt to put a value back into a full cell";
        }
        self.value = some(move value);
    }

    /// Returns true if the cell is empty and false if the cell is full.
    fn is_empty() -> bool {
        self.value.is_none()
    }
}
