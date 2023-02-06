trait T1 {}
trait T2 {}
trait T3 {}
trait T4 {}

impl<B: T2> T1 for Wrapper<B> {}

impl T2 for i32 {}
impl T3 for i32 {}

impl<A: T3> T2 for Burrito<A> {}

struct Wrapper<W> {
    value: W,
}

struct Burrito<F> {
    filling: F,
}

fn want<V: T1>(_x: V) {}

fn example<Q>(q: Q) {
    want(Wrapper { value: Burrito { filling: q } });
    //~^ ERROR the trait bound `Q: T3` is not satisfied [E0277]
}

fn main() {}
