// Some examples of attribute parsing previously missing from the test suite

#![feature(rustc_attrs, stmt_expr_attributes)]

fn foo<const N: u8 = #[rustc_dummy] 11>() {}
//~^ ERROR defaults for generic parameters are not allowed here

fn main() {
    match #[rustc_dummy] 10 { _ => {} }
    let _ = - #[rustc_dummy] 10;
    let _ = || #[rustc_dummy] 10;
    if #[rustc_dummy] true {}
    if let _ = #[rustc_dummy] 10 {}
    for _ in #[rustc_dummy] (0..10) {}
    for (_ in #[rustc_dummy] (0..10)) {}
    //~^ ERROR unexpected parentheses surrounding `for` loop head
    foo::<#[rustc_dummy] 10>();
    //~^ ERROR attributes cannot be applied to generic arguments
    cfg_select! { _ => #[rustc_dummy] 10 }
    //~^ ERROR expected expression, found `#`
}
