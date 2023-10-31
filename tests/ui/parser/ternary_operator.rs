// A good chunk of these errors aren't shown to the user, but are still
// required in the test for it to pass.

fn a() { //~ NOTE this function should return `Result` or `Option` to accept `?`
    let x = 5 > 2 ? true : false;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE cannot use the `?` operator in a function that returns `()`
    //~| NOTE in this expansion of desugaring of operator `?`
}

fn b() { //~ NOTE this function should return `Result` or `Option` to accept `?`
    let x = 5 > 2 ? { true } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE cannot use the `?` operator in a function that returns `()`
    //~| NOTE in this expansion of desugaring of operator `?`
}

fn c() { //~ NOTE this function should return `Result` or `Option` to accept `?`
    let x = 5 > 2 ? f32::MAX : f32::MIN;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE cannot use the `?` operator in a function that returns `()`
    //~| NOTE in this expansion of desugaring of operator `?`
}

fn main() { //~ NOTE this function should return `Result` or `Option` to accept `?`
    let x = 5 > 2 ? { let x = vec![]: Vec<u16>; x } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `:`
    //~| NOTE expected one of `.`, `;`, `?`, `else`, or an operator
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
    //~| NOTE type ascription syntax has been removed, see issue #101728 <https://github.com/rust-lang/rust/issues/101728>
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE cannot use the `?` operator in a function that returns `()`
    //~| NOTE in this expansion of desugaring of operator `?`
}
