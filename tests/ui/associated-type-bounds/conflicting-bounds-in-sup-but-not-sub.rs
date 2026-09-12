//@ check-pass
// The dyn-compatibility check for coherent associated types has
// unintuitive behavior in that a subtrait can be dyn-compatible,
// even though a supertrait is dyn-incompatible.
// This can occur when the subtrait instantiates generics to be
// concrete enough that we can prove that the associated type bounds
// do not conflict.

trait Super<T> {
    type Assoc;
}
// `Sub` is dyn-incompatible due to conflicting associated type bounds,
// since we can't prove that `Super<T>` and `Super<U>` are distinct traits.
trait Sub<T, U>: Super<T, Assoc = u32> + Super<U, Assoc = u64> {}
// `SubSub` is dyn-compatible, since it has associated type bounds on
// `Super<i32>` and `Super<i64>`, which are definitely distinct traits.
trait SubSub: Sub<i32, i64> {}

fn check(_: &dyn SubSub) {}

fn main() {}
