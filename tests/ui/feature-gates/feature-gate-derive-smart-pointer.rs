use std::marker::SmartPointer; //~ ERROR use of unstable library feature 'derive_smart_pointer'

#[derive(SmartPointer)] //~ ERROR use of unstable library feature 'derive_smart_pointer'
#[repr(transparent)]
struct MyPointer<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

fn main() {}
