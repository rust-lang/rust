// Verifies that trait methods with custom receivers (e.g., Arc<dyn Trait>) can
// be called through trait objects.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;

trait Trait1 {
    fn foo(self: Arc<Self>) -> i32;
    fn bar(self: Box<Self>) -> i32;
    fn baz(self: Rc<Self>) -> i32;
    fn qux(self: Pin<&mut Self>) -> i32;
}

struct Type1;

impl Trait1 for Type1 {
    // <Type1 as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo
    fn foo(self: Arc<Self>) -> i32 {
        1
    }
    // <Type1 as Trait1>::bar is transformed into <dyn Trait1 as Trait1>::bar
    fn bar(self: Box<Self>) -> i32 {
        2
    }
    // <Type1 as Trait1>::baz is transformed into <dyn Trait1 as Trait1>::baz
    fn baz(self: Rc<Self>) -> i32 {
        3
    }
    // <Type1 as Trait1>::qux is transformed into <dyn Trait1 as Trait1>::qux
    fn qux(self: Pin<&mut Self>) -> i32 {
        4
    }
}

fn main() {
    let x: Arc<dyn Trait1> = Arc::new(Type1);
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::foo
    assert_eq!(x.foo(), 1);
    let x: Box<dyn Trait1> = Box::new(Type1);
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::bar
    assert_eq!(x.bar(), 2);
    let x: Rc<dyn Trait1> = Rc::new(Type1);
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::baz
    assert_eq!(x.baz(), 3);
    let mut y = Type1;
    let x: Pin<&mut Type1> = Pin::new(&mut y);
    let x: Pin<&mut dyn Trait1> = x;
    // The virtual method call is transformed into <dyn Trait1 as Trait1>::qux
    assert_eq!(x.qux(), 4);
}
