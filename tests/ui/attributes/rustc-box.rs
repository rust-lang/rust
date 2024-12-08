#![feature(rustc_attrs, stmt_expr_attributes)]

fn foo(_: u32, _: u32) {}
fn bar(_: u32) {}

fn main() {
    #[rustc_box]
    Box::new(1); // OK
    #[rustc_box]
    Box::pin(1); //~ ERROR `#[rustc_box]` attribute used incorrectly
    #[rustc_box]
    foo(1, 1); //~ ERROR `#[rustc_box]` attribute used incorrectly
    #[rustc_box]
    bar(1); //~ ERROR `#[rustc_box]` attribute used incorrectly
    #[rustc_box] //~ ERROR `#[rustc_box]` attribute used incorrectly
    #[rustfmt::skip]
    Box::new(1);
}
