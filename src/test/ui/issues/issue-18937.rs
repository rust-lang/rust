// Regression test for #18937.

use std::fmt;

#[derive(Debug)]
struct MyString<'a>(&'a String);

struct B {
    list: Vec<Box<dyn fmt::Debug>>,
}

trait A<'a> {
    fn foo<F>(&mut self, f: F)
        where F: fmt::Debug + 'a,
              Self: Sized;
}

impl<'a> A<'a> for B {
    fn foo<F>(&mut self, f: F) //~ ERROR impl has stricter
        where F: fmt::Debug + 'static,
    {
        self.list.push(Box::new(f));
    }
}

fn main() {
    let mut b = B { list: Vec::new() };

    // Create a borrowed pointer, put it in `b`, then drop what's borrowing it
    let a = "hello".to_string();
    b.foo(MyString(&a));

    // Drop the data which `b` has a reference to
    drop(a);

    // Use the data, probably segfaulting
    for b in b.list.iter() {
        println!("{:?}", b);
    }
}
