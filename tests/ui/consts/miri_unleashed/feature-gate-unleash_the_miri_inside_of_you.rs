// a test demonstrating why we do need to run static const qualification on associated constants
// instead of just checking the final constant

trait Foo<T> {
    const X: T;
}

trait Bar<T, U: Foo<T>> {
    const F: u32 = (U::X, 42).1; //~ ERROR destructor of
}

impl Foo<u32> for () {
    const X: u32 = 42;
}

impl Foo<Vec<u32>> for String {
    const X: Vec<u32> = Vec::new();
}

impl Bar<u32, ()> for () {}
impl Bar<Vec<u32>, String> for String {}

fn main() {
    let x = <() as Bar<u32, ()>>::F;
    let y = <String as Bar<Vec<u32>, String>>::F;
}
