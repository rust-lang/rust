// run-rustfix
struct Foo {
    x: i32,
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("kaboom");
    }
}

fn main() {
    let mut x = Foo { x: -7 };
    x.x = 0;
    println!("{}", x.x);
    x.drop();
    //~^ ERROR E0040
}
