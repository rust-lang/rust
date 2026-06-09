//@ build-fail
//@ dont-require-annotations: NOTE

// a test demonstrating that const qualification cannot prevent monomorphization time errors

trait Foo {
    const X: u32;
}

trait Bar<U: Foo> {
    const F: u32 = 100 / U::X; //~ ERROR attempt to divide `100_u32` by zero
}

impl Foo for () {
    const X: u32 = 42;
}

impl Foo for String {
    const X: u32 = 0;
}

impl Bar<()> for () {}
impl Bar<String> for String {}

fn main() {
    let x = <() as Bar<()>>::F;
    // this test only causes errors due to the line below, so post-monomorphization
    let y = <String as Bar<String>>::F; //~ NOTE constant
}
