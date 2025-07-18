//! Regression test for https://github.com/rust-lang/rust/issues/11192

struct Foo {
    x: isize
}


impl Drop for Foo {
    fn drop(&mut self) {
        println!("drop {}", self.x);
    }
}


fn main() {
    let mut ptr: Box<_> = Box::new(Foo { x: 0 });
    let mut test = |foo: &Foo| {
        println!("access {}", foo.x);
        ptr = Box::new(Foo { x: ptr.x + 1 });
        println!("access {}", foo.x);
    };
    test(&*ptr);
    //~^ ERROR: cannot borrow `*ptr` as immutable
}
