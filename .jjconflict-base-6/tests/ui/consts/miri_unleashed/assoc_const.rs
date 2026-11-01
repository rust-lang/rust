//@ build-fail
//@ compile-flags: -Zunleash-the-miri-inside-of-you

// a test demonstrating why we do need to run static const qualification on associated constants
// instead of just checking the final constant


struct NotConstDestruct;

impl Drop for NotConstDestruct {
    fn drop(&mut self) {}
}

trait Foo<T> {
    const X: T;
}

trait Bar<T, U: Foo<T>> {
    const F: u32 = (U::X, 42).1; //~ ERROR
}

impl Foo<u32> for () {
    const X: u32 = 42;
}
impl Foo<NotConstDestruct> for NotConstDestruct {
    const X: NotConstDestruct = NotConstDestruct;
}

impl Bar<u32, ()> for () {}
impl Bar<NotConstDestruct, NotConstDestruct> for NotConstDestruct {}

fn main() {
    // this is fine, but would have been forbidden by the static checks on `F`
    let x = <() as Bar<u32, ()>>::F;
    // this test only causes errors due to the line below, so post-monomorphization
    let y = <NotConstDestruct as Bar<NotConstDestruct, NotConstDestruct>>::F;
}

//~? WARN skipping const checks
