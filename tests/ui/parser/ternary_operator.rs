// A good chunk of these errors aren't shown to the user, but are still
// required in the test for it to pass.

fn a() {
    let x = 5 > 2 ? true : false;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
}

fn b() {
    let x = 5 > 2 ? { true } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
}

fn c() {
    let x = 5 > 2 ? f32::MAX : f32::MIN;
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
}

fn main() {
    let x = 5 > 2 ? { let x = vec![]: Vec<u16>; x } : { false };
    //~^ ERROR Rust has no ternary operator
    //~| HELP use an `if-else` expression instead
    //~| ERROR expected one of `.`, `;`, `?`, `else`, or an operator, found `:`
    //~| NOTE expected one of `.`, `;`, `?`, `else`, or an operator
    //~| ERROR the `?` operator can only be applied to values that implement `Try` [E0277]
    //~| HELP the trait `Try` is not implemented for `{integer}`
    //~| NOTE type ascription syntax has been removed, see issue #101728 <https://github.com/rust-lang/rust/issues/101728>
    //~| NOTE in this expansion of desugaring of operator `?`
    //~| NOTE the `?` operator cannot be applied to type `{integer}`
    //~| NOTE in this expansion of desugaring of operator `?`
}
