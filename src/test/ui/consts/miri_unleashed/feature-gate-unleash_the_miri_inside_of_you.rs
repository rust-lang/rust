#![allow(const_err)]

// a test demonstrating why we do need to run static const qualification on associated constants
// instead of just checking the final constant

trait Foo<T> {
    const X: T;
}

trait Bar<T, U: Foo<T>> {
    const F: u32 = (U::X, 42).1; //~ ERROR destructors cannot be evaluated at compile-time
}

impl Foo<u32> for () {
    const X: u32 = 42;
}
impl Foo<Vec<u32>> for String {
    const X: Vec<u32> = Vec::new(); //~ ERROR not yet stable as a const fn
}

impl Bar<u32, ()> for () {}
impl Bar<Vec<u32>, String> for String {}

fn main() {
    let x = <() as Bar<u32, ()>>::F;
    let y = <String as Bar<Vec<u32>, String>>::F;
}
