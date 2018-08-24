// Test that a class with a non-copyable field can't be
// copied

#[derive(Debug)]
struct bar {
  x: isize,
}

impl Drop for bar {
    fn drop(&mut self) {}
}

fn bar(x:isize) -> bar {
    bar {
        x: x
    }
}

#[derive(Debug)]
struct foo {
  i: isize,
  j: bar,
}

fn foo(i:isize) -> foo {
    foo {
        i: i,
        j: bar(5)
    }
}

fn main() {
    let x = foo(10);
    let _y = x.clone(); //~ ERROR no method named `clone` found
    println!("{:?}", x);
}
