#[derive(Debug)]
struct Foo {
  i: isize,
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn foo(i:isize) -> Foo {
    Foo {
        i: i
    }
}

fn main() {
    let x = foo(10);
    let _y = x.clone();
    //~^ ERROR no method named `clone` found
    println!("{:?}", x);
}
