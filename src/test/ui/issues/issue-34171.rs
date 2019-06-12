// check-pass

macro_rules! null { ($i:tt) => {} }
macro_rules! apply_null {
    ($i:item) => { null! { $i } }
}

fn main() {
    apply_null!(#[cfg(all())] fn f() {});
}
