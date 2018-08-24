#[derive(Debug)]
struct foo {
  i: isize,
}

impl Drop for foo {
    fn drop(&mut self) {}
}

fn foo(i:isize) -> foo {
    foo {
        i: i
    }
}

fn main() {
    let x = foo(10);
    let _y = x.clone();
    //~^ ERROR no method named `clone` found
    println!("{:?}", x);
}
