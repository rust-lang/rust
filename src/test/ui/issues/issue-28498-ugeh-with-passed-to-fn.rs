// run-pass

// Demonstrate the use of the unguarded escape hatch with a type param in negative position
// to assert that destructor will not access any dead data.
//
// Compare with ui/span/issue28498-reject-lifetime-param.rs

// Demonstrate that a type param in negative position causes dropck to reject code
// that might indirectly access previously dropped value.
//
// Compare with run-pass/issue28498-ugeh-with-passed-to-fn.rs

#![feature(dropck_eyepatch)]

#[derive(Debug)]
struct ScribbleOnDrop(String);

impl Drop for ScribbleOnDrop {
    fn drop(&mut self) {
        self.0 = format!("DROPPED");
    }
}

struct Foo<T>(u32, T, Box<for <'r> fn(&'r T) -> String>);

unsafe impl<#[may_dangle] T> Drop for Foo<T> {
    fn drop(&mut self) {
        // Use of `may_dangle` is sound, because destructor never passes a `self.1`
        // to the callback (in `self.2`) despite having it available.
        println!("Dropping Foo({}, _)", self.0);
    }
}

fn callback(s: & &ScribbleOnDrop) -> String { format!("{:?}", s) }

fn main() {
    let (last_dropped, foo0);
    let (foo1, first_dropped);

    last_dropped = ScribbleOnDrop(format!("last"));
    first_dropped = ScribbleOnDrop(format!("first"));
    foo0 = Foo(0, &last_dropped, Box::new(callback));
    foo1 = Foo(1, &first_dropped, Box::new(callback));

    println!("foo0.1: {:?} foo1.1: {:?}", foo0.1, foo1.1);
}
