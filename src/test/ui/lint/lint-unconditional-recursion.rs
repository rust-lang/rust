#![deny(unconditional_recursion)]

#![allow(dead_code)]
fn foo() { //~ ERROR function cannot return without recursing
    foo();
}

fn bar() {
    if true {
        bar()
    }
}

fn baz() { //~ ERROR function cannot return without recursing
    if true {
        baz()
    } else {
        baz()
    }
}

fn qux() {
    loop {}
}

fn quz() -> bool { //~ ERROR function cannot return without recursing
    if true {
        while quz() {}
        true
    } else {
        loop { quz(); }
    }
}

// Trait method calls.
trait Foo {
    fn bar(&self) { //~ ERROR function cannot return without recursing
        self.bar()
    }
}

impl Foo for Box<dyn Foo + 'static> {
    fn bar(&self) { //~ ERROR function cannot return without recursing
        loop {
            self.bar()
        }
    }
}

// Trait method call with integer fallback after method resolution.
impl Foo for i32 {
    fn bar(&self) { //~ ERROR function cannot return without recursing
        0.bar()
    }
}

impl Foo for u32 {
    fn bar(&self) {
        0.bar()
    }
}

// Trait method calls via paths.
trait Foo2 {
    fn bar(&self) { //~ ERROR function cannot return without recursing
        Foo2::bar(self)
    }
}

impl Foo2 for Box<dyn Foo2 + 'static> {
    fn bar(&self) { //~ ERROR function cannot return without recursing
        loop {
            Foo2::bar(self)
        }
    }
}

struct Baz;
impl Baz {
    // Inherent method call.
    fn qux(&self) { //~ ERROR function cannot return without recursing
        self.qux();
    }

    // Inherent method call via path.
    fn as_ref(&self) -> &Self { //~ ERROR function cannot return without recursing
        Baz::as_ref(self)
    }
}

// Trait method calls to impls via paths.
impl Default for Baz {
    fn default() -> Baz { //~ ERROR function cannot return without recursing
        let x = Default::default();
        x
    }
}

// Overloaded operators.
impl std::ops::Deref for Baz {
    type Target = ();
    fn deref(&self) -> &() { //~ ERROR function cannot return without recursing
        &**self
    }
}

impl std::ops::Index<usize> for Baz {
    type Output = Baz;
    fn index(&self, x: usize) -> &Baz { //~ ERROR function cannot return without recursing
        &self[x]
    }
}

// Overloaded autoderef.
struct Quux;
impl std::ops::Deref for Quux {
    type Target = Baz;
    fn deref(&self) -> &Baz { //~ ERROR function cannot return without recursing
        self.as_ref()
    }
}

fn all_fine() {
    let _f = all_fine;
}

// issue 26333
trait Bar {
    fn method<T: Bar>(&self, x: &T) {
        x.method(x)
    }
}

fn main() {}
