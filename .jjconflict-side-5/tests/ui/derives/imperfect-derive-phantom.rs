// Test that the "imperfect derives" note is emitted when the derive bound
// is genuinely unnecessary (e.g., `PhantomData<T>`).

use std::marker::PhantomData;

#[derive(Clone)]
struct S<T>(PhantomData<T>);

struct X;

fn require_clone<T: Clone>(_t: T) {}

fn main() {
    require_clone(S::<X>(PhantomData));
    //~^ ERROR the trait bound `S<X>: Clone` is not satisfied
}
