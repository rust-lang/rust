struct AssertCopy<T: Copy>(T);

trait Foo {
    type Assoc;
    type Other;
    const X: Self::Other;
    fn my_method() -> Self::Assoc;
}

impl Foo for bool {
    type Assoc = AssertCopy<String>; //~ ERROR: E0277
    type Other = Self::Assoc; //~ ERROR: E0277
    const X: Self::Other = panic!(); //~ ERROR: E0277
    //~^ ERROR: E0277
    fn my_method() -> Self::Assoc {
        panic!();
    }
}

fn main() {}
