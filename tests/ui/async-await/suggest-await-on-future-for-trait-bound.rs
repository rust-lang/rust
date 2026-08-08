//! Regression test for <https://github.com/rust-lang/rust/issues/159484>.
//! When a future is used directly where a trait is expected (e.g. `take_copy(n)`),
//! suggest adding `.await` at the usage site: `take_copy(n.await)`.
//! Also checks that no suggestion is given when `Future::Output` doesn't implement the trait.
//@ edition:2021

#![crate_type = "lib"]

fn take_copy(_: impl Copy) {}

async fn make_number() -> i32 {
    42
}

async fn use_number() {
    let number = make_number();
    take_copy(number);
    //~^ ERROR the trait bound `impl Future<Output = i32>: Copy` is not satisfied
}

async fn make_string() -> String {
    String::new()
}

// String doesn't implement Copy, so no suggestion.
async fn use_string() {
    let string = make_string();
    take_copy(string);
    //~^ ERROR the trait bound `impl Future<Output = String>: Copy` is not satisfied
}
