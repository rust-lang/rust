// issue #36286

struct S<T: Clone> { a: T }

struct NoClone;
type A = S<NoClone>; //~ ERROR the trait bound `NoClone: Clone` is not satisfied

fn main() {
    let s = A { a: NoClone };
}
