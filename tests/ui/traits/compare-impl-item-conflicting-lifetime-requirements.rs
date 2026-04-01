//@ compile-flags: --crate-type=lib
// Regression test for https://github.com/rust-lang/rust/issues/143872

trait Project {
    type Ty;
}

impl Project for &'_ &'static () {
    type Ty = ();
}

trait Trait {
    fn get<'s>(s: &'s str, _: ()) -> &'_ str;
}

impl Trait for () {
    fn get<'s>(s: &'s str, _: <&&'s () as Project>::Ty) -> &'static str {
        //~^ ERROR cannot infer an appropriate lifetime for lifetime parameter 's in generic type due to conflicting requirements
        //~| ERROR mismatched types
        //~| ERROR lifetime may not live long enough
        s
    }
}
