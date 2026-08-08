//@ run-pass

// This is a legitimate use case, where the `dyn Sub` trait object ends up
// having two different "values" for `Assoc`. This is allowed because we know
// that the two values are for `Super<i32>` and `Super<i64>`, which can't
// possibly be the same trait.

trait Super<T> {
    type Assoc;
}

trait Sub: Super<i32, Assoc = u32> + Super<i64, Assoc = u64> {
    fn method(&self) {}
}

fn foo(x: &dyn Sub) {
    x.method();
}

struct Thing;
impl Super<i32> for Thing {
    type Assoc = u32;
}
impl Super<i64> for Thing {
    type Assoc = u64;
}
impl Sub for Thing {}

fn main() {
    foo(&Thing);
}
