//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// This was fixed by lazy norm of param env with the next solver.
// But it regressed again as we switched back to be consistent with
// the old solver. See #158643.

//[next]~^^^^^^^^ ERROR: the trait bound `F: MyFn<i32>` is not satisfied

trait MyFn<T> {
    type Output;
}

trait Callback<A>: MyFn<A, Output = Self::Ret> {
    type Ret;
}

impl<A, F: MyFn<A>> Callback<A> for F {
    type Ret = F::Output;
}

struct Thing;
trait Trait {}
impl Trait for Thing {}

trait ChannelSender {
    type CallbackArg;

    fn autobatch<F>(self) -> impl Trait
    where
        F: Callback<Self::CallbackArg>;
        //[current]~^ ERROR the trait bound `F: Callback<i32>` is not satisfied
}

struct Sender;

impl ChannelSender for Sender {
    type CallbackArg = i32;

    fn autobatch<F>(self) -> impl Trait
    //~^ ERROR the trait bound `F: MyFn<i32>` is not satisfied
    //~| ERROR the trait bound `F: MyFn<i32>` is not satisfied
    //~| ERROR the trait bound `F: MyFn<i32>` is not satisfied
    //~| ERROR the trait bound `F: MyFn<i32>` is not satisfied
    //[next]~| ERROR the trait bound `F: MyFn<i32>` is not satisfied
    where
        F: Callback<Self::CallbackArg>,
        //[current]~^ ERROR the trait bound `F: MyFn<i32>` is not satisfied
        //[current]~| ERROR the trait bound `F: Callback<i32>` is not satisfied
        {
        Thing
    }
}

fn main() {}
