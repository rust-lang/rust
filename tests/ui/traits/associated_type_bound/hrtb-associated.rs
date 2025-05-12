//@ check-pass
//! This test ensures that HRTB (higher-ranked trait bounds) on associated types
//! compile correctly. This was previously rejected by the compiler.
//! Related issue: <https://github.com/rust-lang/rust/issues/34834>

pub trait Provides<'a> {
    type Item;
}

pub trait Selector: for<'a> Provides<'a> {
    type Namespace: PartialEq + for<'a> PartialEq<<Self as Provides<'a>>::Item>;

    fn get_namespace(&self) -> <Self as Provides>::Item;
}

pub struct MySelector;

impl<'a> Provides<'a> for MySelector {
    type Item = &'a str;
}

impl Selector for MySelector {
    type Namespace = String;

    fn get_namespace(&self) -> &str {
        unimplemented!()
    }
}

fn main() {}
