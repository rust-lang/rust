// Check that we get an error when you use `<Self as Get>::Value` in
// the trait definition even if there is no default method.

trait Get {
    type Value;
}

trait Other {
    fn okay<U:Get>(&self, foo: U, bar: <Self as Get>::Value);
    //~^ ERROR E0277
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
