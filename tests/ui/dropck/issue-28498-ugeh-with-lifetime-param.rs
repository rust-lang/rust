//@ run-pass

// Demonstrate the use of the unguarded escape hatch with a lifetime param
// to assert that destructor will not access any dead data.
//
// Compare with ui/span/issue28498-reject-lifetime-param.rs

#![feature(dropck_eyepatch)]

#[derive(Debug)]
struct ScribbleOnDrop(String);

impl Drop for ScribbleOnDrop {
    fn drop(&mut self) {
        self.0 = format!("DROPPED");
    }
}

struct Foo<'a>(u32, &'a ScribbleOnDrop);

unsafe impl<#[may_dangle] 'a> Drop for Foo<'a> {
    fn drop(&mut self) {
        // Use of `may_dangle` is sound, because destructor never accesses `self.1`.
        println!("Dropping Foo({}, _)", self.0);
    }
}

fn main() {
    let (last_dropped, foo0);
    let (foo1, first_dropped);

    last_dropped = ScribbleOnDrop(format!("last"));
    first_dropped = ScribbleOnDrop(format!("first"));
    foo0 = Foo(0, &last_dropped);
    foo1 = Foo(1, &first_dropped);

    println!("foo0.1: {:?} foo1.1: {:?}", foo0.1, foo1.1);
}
