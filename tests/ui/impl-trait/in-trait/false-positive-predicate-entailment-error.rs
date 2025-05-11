//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

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
    //[current]~^ ERROR the trait bound `F: MyFn<i32>` is not satisfied
    //[current]~| ERROR the trait bound `F: MyFn<i32>` is not satisfied
    //[current]~| ERROR the trait bound `F: MyFn<i32>` is not satisfied
    //[current]~| ERROR the trait bound `F: MyFn<i32>` is not satisfied
    where
        F: Callback<Self::CallbackArg>,
        //[current]~^ ERROR the trait bound `F: MyFn<i32>` is not satisfied
        //[current]~| ERROR the trait bound `F: Callback<i32>` is not satisfied
        {
        Thing
    }
}

fn main() {}
