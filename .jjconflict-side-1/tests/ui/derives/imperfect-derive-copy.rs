// Test that the "imperfect derives" note is emitted for Copy when the derive
// bound is genuinely unnecessary (e.g., `PhantomData<T>`).

use std::marker::PhantomData;

#[derive(Copy, Clone)]
struct X<T>(PhantomData<T>);

struct Y; // does not implement Copy

fn require_copy<T: Copy>(_t: T) {}

fn main() {
    require_copy(X::<Y>(PhantomData));
    //~^ ERROR the trait bound `X<Y>: Copy` is not satisfied
}
