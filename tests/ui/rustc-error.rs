#![feature(rustc_attrs)]

#[rustc_error]
fn main() {
    //~^ ERROR fatal error triggered by #[rustc_error]
}
