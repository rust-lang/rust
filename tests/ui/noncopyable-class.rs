// Test that a class with a non-copyable field can't be
// copied

#[derive(Debug)]
struct Bar {
  x: isize,
}

impl Drop for Bar {
    fn drop(&mut self) {}
}

fn bar(x:isize) -> Bar {
    Bar {
        x: x
    }
}

#[derive(Debug)]
struct Foo {
  i: isize,
  j: Bar,
}

fn foo(i:isize) -> Foo {
    Foo {
        i: i,
        j: bar(5)
    }
}

fn main() {
    let x = foo(10);
    let _y = x.clone(); //~ ERROR no method named `clone` found
    println!("{:?}", x);
}
