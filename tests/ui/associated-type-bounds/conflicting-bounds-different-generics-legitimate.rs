//@ run-pass

// Legitimate use case #1:
// The `dyn Sub1` trait object ends up having two different "values" for `Assoc`.
// This is allowed because we know that the two values are for `Super<i32>` and
// `Super<i64>`, which can't possibly be the same trait.

trait Super<T> {
    type Assoc;
}

trait Sub1: Super<i32, Assoc = u32> + Super<i64, Assoc = u64> {
    fn method1(&self) {}
}

fn foo(x: &dyn Sub1) {
    x.method1();
}

// Legitimate use case #2:
// The `dyn Sub2<T, U>` trait object has "values" for `Assoc` specified twice,
// on different generics that might be actually equal types.
// This is allowed because the two values are the same, so there's no conflict.

trait Sub2<T, U>: Super<T, Assoc = u32> + Super<U, Assoc = u32> {
    fn method2(&self) {}
}

fn bar<T, U>(x: &dyn Sub2<T, U>) {
    x.method2();
}

// Actually call the two functions.

struct Thing;
impl Super<i32> for Thing {
    type Assoc = u32;
}
impl Super<i64> for Thing {
    type Assoc = u64;
}
impl Sub1 for Thing {}
impl Sub2<i32, i32> for Thing {}

fn main() {
    foo(&Thing);
    bar(&Thing);
}
