// https://github.com/rust-lang/rust/issues/82329
//@ compile-flags: -Zunpretty=hir,typed
//@ check-pass
//@ edition:2015

pub fn main() {
    if true {
    } else if let Some(a) = Some(3) {
    }
}
