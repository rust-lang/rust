// Check that substitutions given on the self type (here, `A`) carry
// through to NLL.

struct A<'a> { x: &'a u32 }

impl<'a> A<'a> {
    fn new<'b, T>(x: &'a u32, y: T) -> Self {
        Self { x }
    }
}

fn foo<'a>() {
    let v = 22;
    let x = A::<'a>::new(&v, 22);
    //~^ ERROR
}

fn main() {}
