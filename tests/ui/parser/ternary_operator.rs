fn a() {
    let x = 5 > 2 ? true : false;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
}

fn b() {
    let x = 5 > 2 ? { true } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
}

fn c() {
    let x = 5 > 2 ? f32::MAX : f32::MIN;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
}

fn main() {
    let x = 5 > 2 ? { let x = vec![]: Vec<u16>; x } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| expected one of `.`, `;`, `?`, `else`, or an operator, found `:`
        //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| ERROR the `?` operator can only be used in a function that returns `Result` or `Option` (or another type that implements `FromResidual`) [E0277]
    //~| HELP the trait `FromResidual<_>` is not implemented for `()`
}
