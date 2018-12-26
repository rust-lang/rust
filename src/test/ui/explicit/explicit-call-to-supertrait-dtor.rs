struct Foo {
    x: isize
}

trait Bar : Drop {
    fn blah(&self);
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("kaboom");
    }
}

impl Bar for Foo {
    fn blah(&self) {
        self.drop();    //~ ERROR explicit use of destructor method
    }
}

fn main() {
    let x = Foo { x: 3 };
}
