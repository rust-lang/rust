// Demonstrates and records a theoretical regressions / breaking changes caused by the
// introduction of const trait bounds.

// Setting the edition to 2018 since we don't regress `demo! { dyn const }` in Rust <2018.
//@ edition:2018

macro_rules! demo {
    ($ty:ty) => { compile_error!("ty"); };
    (impl $c:ident) => {};
    (dyn $c:ident) => {};
}

demo! { impl const }
//~^ ERROR expected identifier, found `<eof>`

demo! { dyn const }
//~^ ERROR const trait impls are experimental
//~| ERROR expected identifier, found `<eof>`

fn main() {}
