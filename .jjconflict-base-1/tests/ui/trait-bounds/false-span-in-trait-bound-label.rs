// In this test, the span of the trait bound label should point to `1`, not `""`.
// See issue #143336

trait A<T> {
    fn f(self, x: T);
}

fn main() {
    A::f(1, ""); //~ ERROR the trait bound `{integer}: A<_>` is not satisfied [E0277]
}
