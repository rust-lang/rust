#![feature(rustc_attrs)]

macro_rules! null { ($i:tt) => {} }
macro_rules! apply_null {
    ($i:item) => { null! { $i } }
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
    apply_null!(#[cfg(all())] fn f() {});
}
