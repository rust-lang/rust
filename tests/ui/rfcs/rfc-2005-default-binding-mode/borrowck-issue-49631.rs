#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Foo {
}

impl Foo {
    fn get(&self) -> Option<&Result<String, String>> {
        None
    }

    fn mutate(&mut self) { }
}

fn main() {
    let mut foo = Foo { };

    // foo.get() returns type Option<&Result<String, String>>, so
    // using `string` keeps borrow of `foo` alive. Hence calling
    // `foo.mutate()` should be an error.
    while let Some(Ok(string)) = foo.get() {
        foo.mutate();
        //~^ ERROR cannot borrow `foo` as mutable
        println!("foo={:?}", *string);
    }
}
