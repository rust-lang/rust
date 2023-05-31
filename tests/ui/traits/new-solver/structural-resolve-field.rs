// compile-flags: -Ztrait-solver=next
// check-pass

struct Foo {
    x: i32,
}

impl MyDefault for Foo {
    fn my_default() -> Self {
        Self {
            x: 0,
        }
    }
}

trait MyDefault {
    fn my_default() -> Self;
}

impl MyDefault for [Foo; 0]  {
    fn my_default() -> Self {
        []
    }
}
impl MyDefault for [Foo; 1] {
    fn my_default() -> Self {
        [Foo::my_default(); 1]
    }
}

fn main() {
    let mut xs = <[Foo; 1]>::my_default();
    xs[0].x = 1;
    (&mut xs[0]).x = 2;
}
