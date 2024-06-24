use std::marker::SmartPointer; //~ ERROR use of unstable library feature 'derive_smart_pointer'

#[derive(SmartPointer)] //~ ERROR use of unstable library feature 'derive_smart_pointer'
struct MyPointer<'a, #[pointee] T: ?Sized> {
    //~^ ERROR the `#[pointee]` attribute is an experimental feature
    ptr: &'a T,
}

fn main() {}
