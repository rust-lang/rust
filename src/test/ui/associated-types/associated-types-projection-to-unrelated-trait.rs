// run-pass
// Check that we do not get an error when you use `<Self as Get>::Value` in
// the trait definition if there is no default method and for every impl,
// `Self` does implement `Get`.
//
// See also tests associated-types-no-suitable-supertrait
// and associated-types-no-suitable-supertrait-2, which show how small
// variants of the code below can fail.

trait Get {
    type Value;
}

trait Other {
    fn okay<U:Get>(&self, foo: U, bar: <Self as Get>::Value)
        where Self: Get;
}

impl Get for () {
    type Value = f32;
}

impl Get for f64 {
    type Value = u32;
}

impl Other for () {
    fn okay<U:Get>(&self, _foo: U, _bar: <Self as Get>::Value) { }
}

impl Other for f64 {
    fn okay<U:Get>(&self, _foo: U, _bar: <Self as Get>::Value) { }
}

fn main() { }
