struct AssertCopy<T: Copy>(T);

trait Foo {
    type Assoc;
    type Other;
    const X: Self::Other;
    fn my_method() -> Self::Assoc;
}

impl Foo for bool {
    type Assoc = AssertCopy<String>; //~ ERROR: E0277
    type Other = Self::Assoc; // ok
    const X: Self::Other = panic!(); // ok
    fn my_method() -> Self::Assoc { // ok
        panic!();
    }
}

fn main() {}
