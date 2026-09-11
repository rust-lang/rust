//! Regression test for <https://github.com/rust-lang/rust/issues/159484>.
//! When a future is used inside a format macro like `println!("{number}")`, the
//! span is `{number}` (inside the string literal), so `.await` cannot be added
//! directly. Instead, suggest adding an intermediate `let number = number.await;`.
//@ edition:2021
//@ run-rustfix

#![crate_type = "lib"]

async fn make_number() -> i32 {
    42
}

pub async fn println_number_display() {
    let number = make_number();
    println!("{number}");
    //~^ ERROR `impl Future<Output = i32>` doesn't implement `std::fmt::Display`
}

pub async fn println_number_debug() {
    let number = make_number();
    println!("{number:?}");
    //~^ ERROR `impl Future<Output = i32>` doesn't implement `Debug`
}

pub async fn println_number_display_format_spec() {
    let number = make_number();
    println!("{number:.2}");
    //~^ ERROR `impl Future<Output = i32>` doesn't implement `std::fmt::Display`
}
