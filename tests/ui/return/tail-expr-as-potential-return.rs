//@ edition:2018

// > Suggest returning tail expressions that match return type
// >
// > Some newcomers are confused by the behavior of tail expressions,
// > interpreting that "leaving out the `;` makes it the return value".
// > To help them go in the right direction, suggest using return instead
// > when applicable.
// (original commit description for this test)
//
// This test was amended to also serve as a regression test for #92308, where
// this suggestion would not trigger with async functions.

fn main() {
}

fn foo(x: bool) -> Result<f64, i32> {
    if x {
        Err(42)
        //~^ ERROR mismatched types
        //~| HELP you might have meant to return this value
    }
    Ok(42.0)
}

async fn bar(x: bool) -> Result<f64, i32> {
    if x {
        Err(42)
        //~^ ERROR mismatched types
        //~| HELP you might have meant to return this value
    }
    Ok(42.0)
}

trait Identity {
    type Out;
}

impl<T> Identity for T {
    type Out = T;
}

async fn foo2() -> i32 {
    if true {
        1i32
        //~^ ERROR mismatched types
        //~| HELP you might have meant to return this value
    }
    0
}

struct Receiver;
impl Receiver {
    fn generic<T>(self) -> Option<T> {
        None
    }
}
fn method() -> Option<i32> {
    if true {
        Receiver.generic();
        //~^ ERROR type annotations needed
        //~| HELP consider specifying the generic argument
    }

    None
}
